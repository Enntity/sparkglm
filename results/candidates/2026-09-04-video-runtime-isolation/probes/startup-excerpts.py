# SPDX-License-Identifier: Apache-2.0
"""Extract startup evidence without publishing machine paths or whole logs."""
import argparse
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("logs", nargs="+", help="ARM=PATH for each privately retained startup log")
args = parser.parse_args()
patterns = ["Loading weights took", "Model loading took", "Overriding num_gpu_blocks=",
            "GPU KV cache size:", "Graph capturing finished", "init engine (profile,",
            "boot-shape-warmup: 20/20 requests ok", "c4-capacity-warmup:"]
result = {}
for argument in args.logs:
    arm, raw_path = argument.split("=", 1)
    if not re.fullmatch(r"[a-z0-9-]+", arm):
        raise ValueError("Use a public-safe arm identifier")
    lines = []
    for line in Path(raw_path).read_text().splitlines():
        if any(pattern in line for pattern in patterns):
            lines.append(re.sub(r"\x1b\[[0-9;]*m", "", line))
    if not any("c4-capacity-warmup:" in line for line in lines):
        raise ValueError(f"Completed capacity warmup not found: {arm}")
    result[arm] = lines
print(json.dumps({"scope": "Selected startup events only; complete logs retained privately. Reported graph-memory deltas are UMA accounting observations, not physical negative allocations.", "blocks": result}, indent=2))
