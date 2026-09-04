#!/usr/bin/env python3
"""Repeatable tinyGLM promotion gate with correctness and latency checks."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "benchmarks" / "tinyglm_smoke.py"
CASES = (
    {"name": "decode_c1", "concurrency": 1, "prompt_tokens": 128,
     "output_tokens": 256, "stagger_ms": 0},
    {"name": "mixed_c4", "concurrency": 4, "prompt_tokens": 4096,
     "output_tokens": 128, "stagger_ms": 250},
    {"name": "long_c2", "concurrency": 2, "prompt_tokens": 16384,
     "output_tokens": 32, "stagger_ms": 500},
)


def _smoke_module():
    spec = importlib.util.spec_from_file_location("tinyglm_smoke", SMOKE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(runs: list[dict]) -> dict:
    signatures: list[list[str]] = []
    for run in runs:
        signatures.append(
            [item["token_ids_sha256"] for item in run["streams"]]
        )
    return {
        "aggregate_tok_s_median": statistics.median(
            run["aggregate_tok_s"] for run in runs
        ),
        "wall_s_median": statistics.median(run["wall_s"] for run in runs),
        "ttft_s_median": statistics.median(
            item["ttft_s"] for run in runs for item in run["streams"]
        ),
        "signatures": signatures,
        "deterministic": all(value == signatures[0] for value in signatures),
    }


def run_gate(endpoint: str, model: str, repetitions: int, timeout: float) -> dict:
    smoke = _smoke_module()
    cases = []
    for case in CASES:
        runs = [
            smoke.benchmark(
                endpoint=endpoint,
                model=model,
                concurrency=case["concurrency"],
                prompt_tokens=case["prompt_tokens"],
                output_tokens=case["output_tokens"],
                stagger_ms=case["stagger_ms"],
                timeout=timeout,
            )
            for _ in range(repetitions)
        ]
        cases.append({**case, "runs": runs, "summary": summarize(runs)})
    return {
        "schema": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": model,
        "repetitions": repetitions,
        "cases": cases,
    }


def _case_map(report: dict) -> dict[str, dict]:
    return {case["name"]: case for case in report["cases"]}


def validate_report(report: dict) -> list[str]:
    failures: list[str] = []
    for case in report["cases"]:
        name = case["name"]
        if not case["summary"]["deterministic"]:
            failures.append(f"{name}: output is nondeterministic across repetitions")
        for run_index, run in enumerate(case["runs"]):
            if len(run["streams"]) != case["concurrency"]:
                failures.append(f"{name} run {run_index}: stream count differs")
            for stream in run["streams"]:
                if stream["completion_tokens"] != case["output_tokens"]:
                    failures.append(
                        f"{name} run {run_index} stream {stream['stream']}: "
                        "completion token count differs"
                    )
                if stream["token_ids_count"] != case["output_tokens"]:
                    failures.append(
                        f"{name} run {run_index} stream {stream['stream']}: "
                        "token IDs missing from response"
                    )
    return failures


def compare_reports(
    baseline: dict, candidate: dict, max_regression_pct: float
) -> list[str]:
    failures: list[str] = []
    if baseline.get("schema") != candidate.get("schema"):
        failures.append("report schema differs")
        return failures
    allowed = max_regression_pct / 100.0
    base_cases = _case_map(baseline)
    cand_cases = _case_map(candidate)
    if set(base_cases) != set(cand_cases):
        failures.append("case set differs")
        return failures

    for name, base in base_cases.items():
        cand = cand_cases[name]
        b = base["summary"]
        c = cand["summary"]
        if not c["deterministic"]:
            failures.append(f"{name}: candidate output is nondeterministic")
        if b["signatures"][0] != c["signatures"][0]:
            failures.append(f"{name}: generated token IDs changed")
        floor = b["aggregate_tok_s_median"] * (1.0 - allowed)
        if c["aggregate_tok_s_median"] < floor:
            failures.append(
                f"{name}: throughput {c['aggregate_tok_s_median']:.3f} "
                f"is below {floor:.3f} ({max_regression_pct:.1f}% gate)"
            )
        wall_ceiling = b["wall_s_median"] * (1.0 + allowed) + 0.050
        if c["wall_s_median"] > wall_ceiling:
            failures.append(
                f"{name}: wall {c['wall_s_median']:.3f}s exceeds "
                f"{wall_ceiling:.3f}s"
            )
        ttft_ceiling = b["ttft_s_median"] * (1.0 + allowed) + 0.050
        if c["ttft_s_median"] > ttft_ceiling:
            failures.append(
                f"{name}: TTFT {c['ttft_s_median']:.3f}s exceeds "
                f"{ttft_ceiling:.3f}s"
            )
    return failures


def _print_summary(report: dict) -> None:
    for case in report["cases"]:
        summary = case["summary"]
        print(
            f"{case['name']:10s} aggregate="
            f"{summary['aggregate_tok_s_median']:.1f} tok/s "
            f"TTFT={summary['ttft_s_median']:.3f}s "
            f"wall={summary['wall_s_median']:.3f}s "
            f"deterministic={summary['deterministic']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="tinyGLM-5.3-EXL3")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression-pct", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 2:
        raise SystemExit("--repetitions must be at least 2")
    report = run_gate(args.endpoint, args.model, args.repetitions, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report)

    report_failures = validate_report(report)
    if report_failures:
        print("tinyGLM gate: FAIL")
        for failure in report_failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    if args.baseline is None:
        print(f"tinyGLM baseline recorded: {args.output}")
        return
    baseline = json.loads(args.baseline.read_text())
    failures = compare_reports(baseline, report, args.max_regression_pct)
    if failures:
        print("tinyGLM gate: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("tinyGLM gate: PASS")


if __name__ == "__main__":
    main()
