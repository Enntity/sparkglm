#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""matrix-steps.py NAME=DIR...: split matrix walls into first-token time and time per verify step.

A matrix case's wall depends on the text the model writes: the more drafted
tokens the text accepts, the fewer verify steps 400 tokens take. This prints,
for each matrix directory, the first-token times and the stream-seconds spent
per verify step, which do not depend on the text.
"""
import json, pathlib, sys

CASES = ("c1-16k", "c1-32k", "c2-16k", "c2-32k", "c4-16k")
runs = [arg.split("=", 1) for arg in sys.argv[1:]]
total = {}
for case in CASES:
    print(case)
    for name, directory in runs:
        data = json.loads((pathlib.Path(directory) / f"{case}.json").read_text())
        requests = data["requests"]
        steps = sum(r["completion_tokens"] - r["accepted_prediction_tokens"] for r in requests)
        tokens = sum(r["completion_tokens"] for r in requests)
        decode = sum(r["elapsed_s"] - r["ttft_s"] for r in requests)
        entry = total.setdefault(name, [0, 0.0, 0])
        entry[0] += steps; entry[1] += decode; entry[2] += tokens
        print("  %-12s wall %6.2f  first token %s  tokens %d  verify steps %d  stream-s/step %.4f  chars %s" % (
            name, data["summary"]["wall_s"], [round(r["ttft_s"], 2) for r in requests], tokens, steps,
            decode / steps, [r["output_chars"] for r in requests]))
for name, (steps, decode, tokens) in total.items():
    print("%-12s all cases: tokens %d  verify steps %d  stream-seconds per step %.4f  tokens per step %.2f" % (
        name, tokens, steps, decode / steps, tokens / steps))
