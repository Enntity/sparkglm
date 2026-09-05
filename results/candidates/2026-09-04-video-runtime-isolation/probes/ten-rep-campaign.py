# SPDX-License-Identifier: Apache-2.0
"""Continue the fixed-cache comparison as four independently warmed ABBA blocks.

Original diagnostic orchestration. No engine code or permanent config changes.
Run only with authorization to borrow the two serving hosts. The existing first
five-run block is completed and checked before any container is stopped.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def check_replays(output, arm):
    for name in ["warmup"] + [f"r{i}" for i in range(1, 6)]:
        stem = output / f"{arm}-{name}"
        replay = json.loads(stem.with_suffix(".json").read_text())
        work = json.loads(Path(str(stem) + "-work.json").read_text())
        if len(replay["requests"]) != 4 or any(
            request["error"] or request["completion_tokens"] != 400
            for request in replay["requests"]
        ):
            raise RuntimeError(f"Incomplete replay: {stem}")
        if work["arm"] != arm or work["run"] != name:
            raise RuntimeError(f"Mismatched work receipt: {stem}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--head-container", required=True)
    parser.add_argument("--worker-container", required=True)
    parser.add_argument("--first-driver-pid", type=int, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 3600
    while Path(f"/proc/{args.first_driver_pid}").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("First block driver exceeded one-hour deadline")
        time.sleep(5)
    check_replays(args.output, "clean-fixed650")
    original_digest = "sha256:0b17bd9246763d74e2f5e1b79fecdcb6a8ef03e1b8e5823f2d2183ceafb91159"
    rebuilt_digest = "sha256:730b15d4a094131d29c032a74660236151ae67dd33f49dc7bb1e6b6098d1ce66"
    blocks = [
        ("original-fixed650-b1", "sparkglm-vllm:exl3-grouped-prefill-candidate-20260903d", original_digest, "6a49e49e7e6a3226197a2ceefcf217cdf55f751e"),
        ("original-fixed650-b2", "sparkglm-vllm:exl3-grouped-prefill-candidate-20260903d", original_digest, "6a49e49e7e6a3226197a2ceefcf217cdf55f751e"),
        ("clean-fixed650-b2", "sparkglm-release:video-clean", rebuilt_digest, "672df9e4155fa1fa12f6dc63dbd41f0fa7272ab7"),
    ]
    for arm, image, digest, commit in blocks:
        print(json.dumps({"event": "starting", "arm": arm, "utc_epoch": time.time()}), flush=True)
        # Refuse reruns before mutating the serving state.
        if list(args.output.glob(arm + "-*")):
            raise RuntimeError(f"Block already has artifacts: {arm}")
        run(["./start.sh", "stop"], cwd=args.root)
        env = os.environ.copy()
        env.update(IMAGE=image, SKIP_BUILD="1", SKIP_PULL="1", SKIP_OVERLAY_VERIFY="1",
                   SKIP_DOWNLOAD="1", SKIP_SYNC="1", MODEL_REVISION="lloom-local",
                   DFLASH_REVISION="lloom-local", API_HOST="127.0.0.1",
                   GLM53_BOOT_LONG_C4="1", EXTRA_ARGS="--num-gpu-blocks-override 650")
        with (args.output / f"{arm}-start.log").open("x") as log:
            startup = subprocess.Popen(["./start.sh", "start"], cwd=args.root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(90):
                    state = subprocess.run(["docker", "inspect", args.head_container, "--format", "{{.State.Running}}"], capture_output=True, text=True)
                    if state.returncode == 0 and state.stdout.strip() == "true":
                        break
                    if startup.poll() is not None:
                        raise RuntimeError(f"Startup exited early: {arm}")
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Container did not start: {arm}")
                run(["docker", "update", "--cpuset-cpus", "5-8,15-19", args.head_container])
                run(["ssh", args.worker, "docker", "update", "--cpuset-cpus", "5-8,15-19", args.worker_container])
                if startup.wait(timeout=1500) != 0:
                    raise RuntimeError(f"Startup/warmup failed: {arm}")
            except BaseException:
                if startup.poll() is None:
                    startup.terminate()
                raise
        identities = []
        for prefix, container in [([], args.head_container), (["ssh", args.worker], args.worker_container)]:
            identity = run(prefix + ["docker", "inspect", "--format", "{{.Image}}", container], capture_output=True).stdout.strip()
            cpuset = run(prefix + ["docker", "inspect", "--format", "{{.HostConfig.CpusetCpus}}", container], capture_output=True).stdout.strip()
            if identity != digest or cpuset != "5-8,15-19":
                raise RuntimeError(f"Artifact or CPU placement mismatch: {arm}")
            identities.append({"image_digest": identity, "cpuset": cpuset})
        with urllib.request.urlopen("http://127.0.0.1:8890/metrics", timeout=15) as response:
            metrics = response.read().decode()
        cache = [line for line in metrics.splitlines() if line.startswith("vllm:cache_config_info")]
        if len(cache) != 1 or 'num_gpu_blocks="650"' not in cache[0] or 'kv_cache_size_tokens="1132404"' not in cache[0]:
            raise RuntimeError(f"Cache shape mismatch: {cache}")
        receipt = {"arm": arm, "ranks": identities, "cache_config": cache,
                   "engine_commit": commit, "utc_epoch": time.time(), "retained_runs": 5,
                   "discarded_exact_replays": 1}
        (args.output / f"{arm}-startup.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with (args.output / f"{arm}-campaign.log").open("x") as log:
            run(["python3", str(args.probe), "--root", str(args.root), "--output", str(args.output),
                 "--arm", arm, "--engine-commit", commit, "--runs", "5"], stdout=log, stderr=subprocess.STDOUT)
        check_replays(args.output, arm)
        print(json.dumps({"event": "complete", "arm": arm, "utc_epoch": time.time()}), flush=True)
    print("All four warmed blocks complete; no default or routing changes made.", flush=True)


if __name__ == "__main__":
    main()
