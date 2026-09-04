#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Seeded GLM-5.3 DFlash decode comparison for OpenAI-compatible servers.

The same prompt bytes and sampling parameters can be sent to Atlas and vLLM.
Per-trial response hashes, speculative acceptance, TTFT, and decode rate make
random-trajectory differences visible instead of folding them into one max TPS
number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request


def request_json(url: str, payload: dict, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def token_count(base_url: str, model: str, prompt: str, timeout_s: float) -> int:
    result = request_json(
        base_url.removesuffix("/v1") + "/tokenize",
        {"model": model, "prompt": prompt},
        timeout_s,
    )
    return int(result["count"])


def fixed_prompt(base_url: str, model: str, target: int, timeout_s: float) -> str:
    unit = "benchmark context datum "
    prompt = "seeded GLM DFlash parity fixture " + unit * max(1, target // 3)
    while token_count(base_url, model, prompt, timeout_s) < target:
        missing = target - token_count(base_url, model, prompt, timeout_s)
        prompt += unit * max(1, missing // 3)
    return prompt + "\nReturn exactly 128 numbered lowercase English words, then stop."


def stream_trial(
    base_url: str,
    model: str,
    prompt: str,
    seed: int,
    output_tokens: int,
    timeout_s: float,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 0,
        "min_p": 0.0,
        "top_n_sigma": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        # Mia's template reads `thinking`; Atlas reads `enable_thinking`.
        # Supplying both false gives the same direct-answer template contract.
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        "repetition_detection": {
            "min_pattern_size": 1,
            "max_pattern_size": 64,
            "min_count": 1_000_000,
        },
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    usage: dict = {}
    output: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                visible = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                if visible:
                    first = first or time.perf_counter()
                    output.append(visible)
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    first = first or finished
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    accepted = int(details.get("accepted_prediction_tokens") or 0)
    rejected = int(details.get("rejected_prediction_tokens") or 0)
    text = "".join(output)
    return {
        "seed": seed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion,
        "accepted_prediction_tokens": accepted,
        "rejected_prediction_tokens": rejected,
        "acceptance": accepted / max(1, accepted + rejected),
        "ttft_ms": (first - started) * 1000.0,
        "decode_tok_s": max(0, completion - 1) / max(0.001, finished - first),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_preview": text[:120],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()

    prompt = fixed_prompt(
        args.base_url, args.model, args.prompt_tokens, args.timeout_s
    )
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    rows = []
    for seed in range(args.seed_start, args.seed_start + args.trials):
        row = stream_trial(
            args.base_url,
            args.model,
            prompt,
            seed,
            args.output_tokens,
            args.timeout_s,
        )
        rows.append(row)
        print(json.dumps({"prompt_sha256": prompt_sha, **row}, sort_keys=True))

    print(
        json.dumps(
            {
                "summary": {
                    "prompt_sha256": prompt_sha,
                    "trials": len(rows),
                    "median_decode_tok_s": statistics.median(
                        row["decode_tok_s"] for row in rows
                    ),
                    "median_acceptance": statistics.median(
                        row["acceptance"] for row in rows
                    ),
                    "median_ttft_ms": statistics.median(
                        row["ttft_ms"] for row in rows
                    ),
                }
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
