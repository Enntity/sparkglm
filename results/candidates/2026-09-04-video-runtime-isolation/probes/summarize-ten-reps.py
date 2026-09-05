# SPDX-License-Identifier: Apache-2.0
"""Summarize every retained ABBA replay; no outlier filtering or engine changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics as stats


ARMS = ["clean-fixed650", "original-fixed650-b1", "original-fixed650-b2", "clean-fixed650-b2"]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {"n": len(values), "mean": stats.mean(values), "median": stats.median(values),
            "min": min(values), "max": max(values), "sample_sd": stats.stdev(values)}


def counter(receipt, metric):
    values = [v for k, v in receipt["counter_deltas"].items() if k.startswith(metric + "{")]
    if not values:
        raise ValueError(f"Required counter unavailable: {metric}")
    return sum(values)


def collect(root, arm, name):
    replay = json.loads((root / f"{arm}-{name}.json").read_text())
    work = json.loads((root / f"{arm}-{name}-work.json").read_text())
    requests = replay["requests"]
    if len(requests) != 4 or any(r["error"] or r["completion_tokens"] != 400 for r in requests):
        raise ValueError(f"Incomplete run retained; cannot report successful campaign: {arm}/{name}")
    if [r["prompt_tokens"] for r in requests] != [15807, 15810, 15810, 15809]:
        raise ValueError(f"Input token-count mismatch: {arm}/{name}")
    gaps = [b["t"] - a["t"] for r in requests for a, b in zip(r["events"], r["events"][1:])]
    hashes = [hashlib.sha256("".join(e["text"] for e in r["events"]).encode()).hexdigest() for r in requests]
    gpu = {}
    for field, index in [("sm_mhz", 0), ("temperature_c", 2), ("utilization_pct", 3), ("power_w", 4)]:
        values = []
        for sample in work["gpu_samples"]:
            try:
                value = float(sample["csv"].split(",")[index].strip())
            except (IndexError, ValueError):
                continue
            if math.isfinite(value) and sample["returncode"] == 0:
                values.append(value)
        gpu[field] = distribution(values) if len(values) > 1 else {"n": len(values)}
    return {"arm": arm, "run": name, "wall_s": replay["wall_s"],
            "aggregate_decode_tok_s": replay["aggregate_decode_tok_s"],
            "ttft_s": [r["ttft_s"] for r in requests],
            "request_ended_s": [r["ended_s"] for r in requests],
            "event_gap_p95_s": percentile(gaps, .95), "event_gap_max_s": max(gaps),
            "draft_request_rounds": counter(work, "vllm:spec_decode_num_drafts_total"),
            "accepted_draft_tokens": counter(work, "vllm:spec_decode_num_accepted_tokens_total"),
            "prefix_hits": counter(work, "vllm:prefix_cache_hits_total"),
            "preemptions": counter(work, "vllm:num_preemptions_total"),
            "delivered_tokens": sum(r["completion_tokens"] for r in requests),
            "output_sha256": hashes, "head_gpu_samples": gpu}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    args = parser.parse_args()
    startups = {arm: json.loads((args.raw / f"{arm}-startup.json").read_text()) for arm in ARMS}
    expected_digests = {
        "clean": "sha256:730b15d4a094131d29c032a74660236151ae67dd33f49dc7bb1e6b6098d1ce66",
        "original": "sha256:0b17bd9246763d74e2f5e1b79fecdcb6a8ef03e1b8e5823f2d2183ceafb91159",
    }
    for arm, receipt in startups.items():
        if len(receipt["ranks"]) != 2 or any(
            rank["image_digest"] != expected_digests[arm.split("-")[0]]
            or rank["cpuset"] != "5-8,15-19" for rank in receipt["ranks"]
        ):
            raise ValueError(f"Startup identity mismatch: {arm}")
        if len(receipt["cache_config"]) != 1 or any(
            expected not in receipt["cache_config"][0]
            for expected in ['num_gpu_blocks="650"', 'num_gpu_blocks_override="650"', 'kv_cache_size_tokens="1132404"']
        ):
            raise ValueError(f"Startup cache geometry mismatch: {arm}")
    blocks = {arm: [collect(args.raw, arm, f"r{i}") for i in range(1, 6)] for arm in ARMS}
    warmups = [collect(args.raw, arm, "warmup") for arm in ARMS]
    metrics = ["wall_s", "aggregate_decode_tok_s", "event_gap_p95_s", "event_gap_max_s", "draft_request_rounds"]
    images = {"rebuilt": blocks[ARMS[0]] + blocks[ARMS[3]],
              "preserved": blocks[ARMS[1]] + blocks[ARMS[2]]}
    image_summary = {image: {metric: distribution([r[metric] for r in runs]) for metric in metrics}
                     for image, runs in images.items()}
    block_summary = {arm: {metric: distribution([r[metric] for r in runs]) for metric in metrics}
                     for arm, runs in blocks.items()}
    contrasts = []
    for candidate, baseline in [(ARMS[0], ARMS[1]), (ARMS[3], ARMS[2])]:
        contrasts.append({"rebuilt_block": candidate, "preserved_block": baseline,
                          "mean_change_pct": {m: 100 * (block_summary[candidate][m]["mean"] / block_summary[baseline][m]["mean"] - 1) for m in metrics}})
    rng = random.Random(20260904)
    conditional_intervals = {}
    for metric in ["wall_s", "aggregate_decode_tok_s"]:
        differences = []
        for _ in range(20000):
            sampled = {arm: stats.mean(r[metric] for r in rng.choices(runs, k=5)) for arm, runs in blocks.items()}
            a = (sampled[ARMS[0]] + sampled[ARMS[3]]) / 2
            b = (sampled[ARMS[1]] + sampled[ARMS[2]]) / 2
            differences.append(100 * (a / b - 1))
        conditional_intervals[metric] = {"mean_change_pct": 100 * (image_summary["rebuilt"][metric]["mean"] / image_summary["preserved"][metric]["mean"] - 1),
                                         "conditional_95pct_bootstrap_interval_pct": [percentile(differences, .025), percentile(differences, .975)]}
    result = {"schema": "sparkglm.diagnostic-ten-reps/v1", "order": ARMS,
              "primary_metric": "wall_s", "startups": startups,
              "retained": blocks, "discarded_warmups": warmups,
              "images": image_summary, "blocks": block_summary, "reversed_order_contrasts": contrasts,
              "conditional_intervals": conditional_intervals,
              "interval_method": "20000 stratified within-block bootstrap resamples, five observations per block, seed 20260904. Conditions on these four startups; ignores serial dependence and uncertainty across startups. Not a startup-independent confidence interval or equivalence test.",
              "limitations": ["Only two independent startups per image; repeated requests are clustered within startups.",
                              "ABBA balances a linear time trend, not arbitrary time effects.",
                              "One C4 posted-video workload, not the full G3/G4 matrix.",
                              "Text hashes are transparency checks, not a quality evaluation."]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
