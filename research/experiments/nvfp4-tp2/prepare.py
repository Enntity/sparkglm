#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the copied TP2 recipe without contacting hosts or starting processes.

Only --execute starts this node's rank, after checking local artifacts.
Run separately on each Spark; this tool never SSHes or stops other services.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_weights(path: Path, model: dict) -> None:
    if path.name != model["revision"]:
        raise ValueError("model directory must end in the pinned revision")
    config = json.loads((path / "config.json").read_text())
    if config_digest(config) != model["config_sha256"]:
        raise ValueError("checkpoint configuration differs from the pinned publisher configuration")
    index = path / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not shards:
        raise ValueError("empty checkpoint shard index")
    for shard in shards:
        if Path(shard).name != shard or not (path / shard).is_file() or (path / shard).stat().st_size == 0:
            raise ValueError("missing, empty, or invalid checkpoint shard: " + shard)


def plan(args: argparse.Namespace) -> dict:
    lock = json.loads((ROOT / "candidate.json").read_text())
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        raise ValueError("--image-id must be the locally built derivative's immutable sha256 image ID")
    for value in (args.head_ip, args.worker_ip):
        ipaddress.IPv4Address(value)
    if args.head_ip == args.worker_ip:
        raise ValueError("head and worker addresses must differ")
    for value in (args.interface, args.hca):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
            raise ValueError("invalid network interface or HCA")
    for path in (args.models, args.cache):
        if not path.is_absolute() or any(c in str(path) for c in "\n\r,"):
            raise ValueError("model/cache roots must be absolute paths without commas or newlines")
    env = dict(lock["profile"])
    # Network/runtime settings adapted from the same upstream launch_cluster.
    env.update({
        "HEAD_IP": args.head_ip, "API_HOST": "127.0.0.1",
        "VLLM_HOST_IP": args.head_ip if args.rank == 0 else args.worker_ip,
        "NCCL_SOCKET_IFNAME": args.interface, "GLOO_SOCKET_IFNAME": args.interface,
        "NCCL_IB_HCA": args.hca, "NCCL_IB_DISABLE": "0", "NCCL_NET": "IB",
        "NCCL_IB_ROCE_VERSION_NUM": "2", "NCCL_IB_GID_INDEX": str(args.gid_index),
        "NCCL_NET_PLUGIN": "none", "NCCL_NVLS_ENABLE": "0", "NCCL_CUMEM_ENABLE": "0",
        "NCCL_IB_MERGE_NICS": "0", "NCCL_CROSS_NIC": "0", "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_DEBUG": "INFO", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "VLLM_CACHE_ROOT": "/root/.cache/vllm", "TORCH_CUDA_ARCH_LIST": "12.1a",
        "FLASHINFER_CUDA_ARCH_LIST": "12.1a", "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
    })
    role = "head" if args.rank == 0 else "worker"
    command = ["docker", "run", "-d", "--name", "sparkglm-nvfp4-" + role,
               "--restart", "no", "--gpus", "all", "--network", "host", "--ipc=host",
               "--shm-size", "32g", "--stop-timeout", "60", "--device", "/dev/infiniband",
               "--cap-add", "IPC_LOCK", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
    downloads, mounts = [], {}
    for key, model in lock["models"].items():
        path = args.models / key / model["revision"]
        mounts[key] = str(path)
        downloads.append(["hf", "download", model["repository"], "--revision", model["revision"], "--local-dir", str(path)])
        command += ["--mount", f"type=bind,src={path},dst=/models/{key},readonly"]
    # A separate compiler cache prevents candidate artifacts polluting EXL3's cache.
    command += ["--mount", f"type=bind,src={args.cache},dst=/root/.cache/vllm"]
    for key, value in sorted(env.items()):
        command += ["-e", key + "=" + value]
    command += ["--entrypoint", "bash", args.image_id, "/opt/sparkglm-nvfp4/" + role + ".sh"]
    return {"status": "prepared-unqualified", "rank": args.rank, "upstream": lock["upstream"],
            "models": lock["models"], "mounts": mounts, "environment": env,
            "download_commands": downloads, "docker_argv": command,
            "docker_command": shlex.join(command)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--head-ip", required=True)
    parser.add_argument("--worker-ip", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--hca", required=True)
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.gid_index < 0:
        parser.error("GID index cannot be negative")
    try:
        result = plan(args)
        if args.execute:
            for key, model in result["models"].items():
                validate_weights(Path(result["mounts"][key]), model)
            info = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image_id], text=True))[0]
            if info["Id"] != args.image_id or info["Architecture"] != "arm64" or info["Os"] != "linux":
                raise ValueError("image identity/platform mismatch")
            if info.get("Config", {}).get("Labels", {}).get("ai.enntity.sparkglm.experiment") != "nvfp4-tp2-unqualified":
                raise ValueError("image is not the prepared NVFP4 derivative")
            args.cache.mkdir(parents=True, exist_ok=True)
            os.execvp("docker", result["docker_argv"])
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
