#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture non-secret server identity alongside a full-model qualification arm."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = (
    "IMAGE", "SPEC_METHOD", "DFLASH_REVISION", "MTP_TOKENS",
    "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
    "GPU_MEM_UTIL", "KV_CACHE_DTYPE", "LANGUAGE_MODEL_ONLY",
    "EXL3_FUSED_MOE", "EXL3_FAT_KERNEL", "EXL3_FAT_TILE_M",
    "EXL3_FAT_PAIR", "EXL3_FAT_FUSED_ACT", "EXL3_GROUPED_PREFILL_K4",
    "EXL3_DECODE_COOP_K4", "GLM53_MIXED_PREFILL_CHUNK",
    "GLM53_INDEXER_WORKSPACE", "GLM53_SPINWAIT_MS",
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/models", headers=headers
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        models = json.loads(response.read())
    digests = [
        value.strip()
        for value in os.environ.get("SPARKGLM_IMAGE_DIGESTS", "").split(",")
        if value.strip()
    ]
    payload = {
        "schema": "sparkglm.server-manifest/v1",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_revision": git("rev-parse", "HEAD"),
        "working_tree_clean": not bool(git("status", "--porcelain")),
        "base_url": args.base_url,
        "served_model": args.model,
        "model_revision": os.environ.get("SPARKGLM_MODEL_REVISION"),
        "image_digests": digests,
        "configuration": {
            key: os.environ[key] for key in CONFIG_KEYS if key in os.environ
        },
        "models_response": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
