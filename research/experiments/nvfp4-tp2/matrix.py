#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print commands for the existing AGPL benchmark; never run endpoints here."""
import argparse
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[3]


def commands(url, model, arm, output):
    harness = str(ROOT / "benchmarks/staggered_openai.py")
    cases = [("16k-c1", 1, "16384", "400"), ("32k-c1", 1, "32768", "400"),
             ("64k-c1", 1, "65536", "400"), ("16k-c2", 2, "16384,16384", "400,400"),
             ("32k-c2", 2, "32768,32768", "400,400"),
             ("16k-c4", 4, "16384,16384,16384,16384", "400,400,400,400"),
             ("decode-plus-32k", 2, "512,32768", "1024,400")]
    for repetition in range(4):  # repeat zero is discarded warmup
        for name, concurrency, prompts, outputs in cases:
            # Same salt across arms; unique first-prefix salt across cases/repetitions.
            argv = ["python3", harness, "--base-url", url, "--model", model,
                    "--concurrency", str(concurrency), "--stagger-ms", "5000",
                    "--exact-prompt-tokens", "--prompt-token-list", prompts,
                    "--output-token-list", outputs, "--min-output-token-list", outputs,
                    "--prompt-salt", f"nvfp4-comparison-{name}-r{repetition}", "--timeout-s", "1800"]
            yield shlex.join(argv) + " > " + shlex.quote(str(output / f"{arm}-{name}-r{repetition}.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True, choices=("exl3", "nvfp4"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print("# Create the output directory before running. Discard r0; alternate arms across retained pairs.")
    print("\n".join(commands(args.base_url, args.model, args.arm, args.output)))
