#!/usr/bin/env python3
"""Focused tinyGLM gate for medium/large C1-C2 prefill changes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tinyglm_smoke", ROOT / "benchmarks" / "tinyglm_smoke.py"
)
assert SPEC and SPEC.loader
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)

CASES = (
    ("medium_c1", 1, 4096, 0),
    ("large_c1", 1, 16384, 0),
    ("large_c2_stagger", 2, 16384, 500),
)


def summarize(runs: list[dict]) -> dict:
    return {
        "wall_s_median": statistics.median(run["wall_s"] for run in runs),
        "ttft_s_median": statistics.median(
            stream["ttft_s"] for run in runs for stream in run["streams"]
        ),
        "signatures": [
            [stream["token_ids_sha256"] for stream in run["streams"]]
            for run in runs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8890")
    parser.add_argument("--model", default="tinyGLM-5.3-EXL3")
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")

    report = {"schema": 1, "cases": []}
    for case_index, (name, concurrency, prompt_tokens, stagger_ms) in enumerate(CASES):
        runs = [
            SMOKE.benchmark(
                endpoint=args.endpoint,
                model=args.model,
                concurrency=concurrency,
                prompt_tokens=prompt_tokens,
                output_tokens=1,
                stagger_ms=stagger_ms,
                timeout=180.0,
                # Disjoint seed ranges keep later cases from reusing an APC
                # entry created by an earlier case in this same server.
                prompt_seed=args.seed_base + case_index * 40 + repetition + 1,
            )
            for repetition in range(args.repetitions)
        ]
        summary = summarize(runs)
        report["cases"].append(
            {
                "name": name,
                "concurrency": concurrency,
                "prompt_tokens": prompt_tokens,
                "stagger_ms": stagger_ms,
                "runs": runs,
                "summary": summary,
            }
        )
        print(
            f"{name:16s} TTFT={summary['ttft_s_median']:.4f}s "
            f"wall={summary['wall_s_median']:.4f}s"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        base_cases = {case["name"]: case for case in baseline["cases"]}
        print("comparison")
        for candidate in report["cases"]:
            base = base_cases[candidate["name"]]["summary"]
            current = candidate["summary"]
            ttft_pct = (base["ttft_s_median"] / current["ttft_s_median"] - 1) * 100
            wall_pct = (base["wall_s_median"] / current["wall_s_median"] - 1) * 100
            same = base["signatures"][0] == current["signatures"][0]
            print(
                f"{candidate['name']:16s} TTFT={ttft_pct:+.1f}% "
                f"wall={wall_pct:+.1f}% same_tokens={same}"
            )
            if not same:
                raise SystemExit(f"{candidate['name']}: output token IDs changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
