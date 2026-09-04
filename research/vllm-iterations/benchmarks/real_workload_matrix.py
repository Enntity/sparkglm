#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Run the fixed medium/large C1/C2 GLM-5.3 appliance comparison.

The generated prompts are stable across engines and salted per case so an
earlier case cannot create an accidental prefix-cache hit for a later case.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


CASES = {
    "medium-c1": (1, [8000]),
    "medium-c2": (2, [8000, 8000]),
    "large-c1": (1, [16000]),
    "large-c2": (2, [16000, 16000]),
}


def compact_result(tag: str, case_name: str, raw: dict) -> dict:
    requests = raw["requests"]
    summary = raw["summary"]
    required_request_metrics = (
        "effective_prefill_tok_s",
        "decode_tok_s",
    )
    required_summary_metrics = (
        "aggregate_effective_prefill_tok_s",
        "decode_tok_s_sum",
        "aggregate_decode_tok_s",
    )
    throughput_sample_complete = all(
        request.get(metric) is not None
        for request in requests
        for metric in required_request_metrics
    ) and all(summary.get(metric) is not None for metric in required_summary_metrics)
    isolation_complete = all(
        request.get("own_request_marker") is True
        and not request.get("foreign_request_markers")
        for request in requests
    )
    return {
        "tag": tag,
        "case": case_name,
        "prompt_tokens": [request["prompt_tokens"] for request in requests],
        "prompt_sha256": [request["prompt_sha256"] for request in requests],
        "completion_tokens": [request["completion_tokens"] for request in requests],
        "accepted_prediction_tokens": [
            request["accepted_prediction_tokens"] for request in requests
        ],
        "rejected_prediction_tokens": [
            request["rejected_prediction_tokens"] for request in requests
        ],
        "completed": [request["reached_requested_completion_tokens"] for request in requests],
        "ttft_s": [request["ttft_s"] for request in requests],
        "effective_prefill_tok_s": [
            request["effective_prefill_tok_s"] for request in requests
        ],
        "decode_tok_s": [request["decode_tok_s"] for request in requests],
        "aggregate_effective_prefill_tok_s": summary[
            "aggregate_effective_prefill_tok_s"
        ],
        "decode_tok_s_sum": summary["decode_tok_s_sum"],
        "aggregate_decode_tok_s": summary["aggregate_decode_tok_s"],
        "throughput_sample_complete": throughput_sample_complete,
        "isolation_complete": isolation_complete,
        "own_request_marker": [request["own_request_marker"] for request in requests],
        "foreign_request_markers": [
            request["foreign_request_markers"] for request in requests
        ],
        "output_preview": [request["output_preview"] for request in requests],
        "finish_reason": [request["finish_reason"] for request in requests],
        "error": [request["error"] for request in requests],
        "max_inter_token_gap_s": [request["max_inter_token_gap_s"] for request in requests],
        "output_sha256": [request["output_sha256"] for request in requests],
        "wall_s": summary["wall_s"],
        "successful": summary["successful"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--tag", default="engine")
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help=f"comma-separated subset of: {','.join(CASES)}",
    )
    parser.add_argument("--stagger-ms", type=int, default=5000)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--salt-variant",
        default="v1",
        help="change between repeated runs to prevent a same-case prefix-cache hit",
    )
    args = parser.parse_args()

    selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = [name for name in selected if name not in CASES]
    if unknown:
        parser.error(f"unknown cases: {','.join(unknown)}")
    if not selected:
        parser.error("at least one case is required")

    harness = pathlib.Path(__file__).with_name("staggered_openai.py")
    common = [
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--timeout-s",
        str(args.timeout_s),
    ]
    if args.api_key:
        common.extend(["--api-key", args.api_key])

    if not args.skip_warmup:
        warmup = subprocess.run(
            [
                sys.executable,
                str(harness),
                *common,
                "--concurrency",
                "1",
                "--stagger-ms",
                "0",
                "--prompt-style",
                "mia-structured",
                "--output-tokens",
                "64",
                "--min-output-tokens",
                "64",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_s + 60,
        )
        if warmup.returncode != 0:
            print(warmup.stderr, file=sys.stderr, end="")
            print(warmup.stdout, file=sys.stderr, end="")
            return warmup.returncode

    rows: list[dict] = []
    for case_name in selected:
        concurrency, prompt_sizes = CASES[case_name]
        output_sizes = [args.output_tokens] * concurrency
        command = [
            sys.executable,
            str(harness),
            *common,
            "--concurrency",
            str(concurrency),
            "--stagger-ms",
            str(args.stagger_ms if concurrency > 1 else 0),
            "--prompt-style",
            "isolation",
            "--prompt-token-list",
            ",".join(map(str, prompt_sizes)),
            "--output-token-list",
            ",".join(map(str, output_sizes)),
            "--min-output-token-list",
            ",".join(map(str, output_sizes)),
            "--prompt-salt",
            f"glm53-real-{args.salt_variant}:{case_name}",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_s + 60,
        )
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr, end="")
            print(completed.stdout, file=sys.stderr, end="")
            return completed.returncode
        raw = json.loads(completed.stdout.strip().splitlines()[-1])
        row = compact_result(args.tag, case_name, raw)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    print(json.dumps({"tag": args.tag, "matrix": rows}, sort_keys=True))
    return (
        0
        if all(
            all(row["completed"])
            and row["throughput_sample_complete"]
            and row["isolation_complete"]
            for row in rows
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
