#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Dependency-free staggered-load probe for an OpenAI-compatible endpoint.

The probe measures the user-visible symptom phase interleaving is meant to
fix: request B arrives while request A is decoding or prefilling. It records
TTFT and completion rate per request and emits one machine-readable JSON row.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class Result:
    request_id: int
    scheduled_s: float
    started_s: float
    ttft_s: float | None
    elapsed_s: float
    completion_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int
    output_chars: int
    output_sha256: str
    output_preview: str
    own_request_marker: bool
    foreign_request_markers: list[int]
    visible_events: int
    inter_token_gap_p95_s: float | None
    max_inter_token_gap_s: float | None
    finish_reason: str | None
    error: str | None

    @property
    def decode_tok_s(self) -> float | None:
        if self.ttft_s is None or self.completion_tokens <= 1:
            return None
        decode_s = self.elapsed_s - self.ttft_s
        return (self.completion_tokens - 1) / decode_s if decode_s > 0 else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def isolation_marker(request_id: int) -> str:
    seed = f"atlas-glm53-isolation-v1:{request_id}".encode()
    return "GLMISO-" + hashlib.sha256(seed).hexdigest()[:20].upper()


def make_prompt(request_id: int, approximate_tokens: int) -> str:
    marker = isolation_marker(request_id)
    prefix = (
        f"Request {request_id}. Read the following numbered facts, then answer "
        f"with a concise summary that repeats the isolation marker {marker} "
        "exactly once and does not omit the final fact.\n"
    )
    # Whitespace-token count is deliberately approximate. The result records
    # the requested size so comparisons use identical bytes across recipes.
    facts = [f"fact-{request_id}-{i}: alpha beta gamma delta" for i in range(max(1, approximate_tokens // 6))]
    return prefix + "\n".join(facts)


def benchmark_prompt(request_id: int, args: argparse.Namespace) -> str:
    if args.prompt_style == "mia-structured":
        return "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
    if args.prompt_style == "mia-prose":
        return (
            "Write a detailed step-by-step explanation of how a hash map works, "
            "including collision handling, resizing, and time complexity. Be thorough."
        )
    return make_prompt(request_id, args.prompt_tokens)


def stream_one(
    request_id: int,
    scheduled_s: float,
    epoch: float,
    args: argparse.Namespace,
) -> Result:
    delay = epoch + scheduled_s - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    started = time.monotonic()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": benchmark_prompt(request_id, args)}],
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "top_p": 1,
        # Mia's template reads `thinking`; Atlas reads `enable_thinking`.
        # Supplying both makes the rendered prompt contract identical.
        "chat_template_kwargs": {
            "thinking": args.enable_thinking,
            "enable_thinking": args.enable_thinking,
        },
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.min_output_tokens:
        payload["min_tokens"] = args.min_output_tokens
    if args.disable_loop_watchdog:
        # Mia's vLLM benchmark has no Atlas content-loop watchdog.  Structured
        # counting is intentionally repetitive, so outrank the production
        # safeguard for only this request rather than restarting the service.
        payload["repetition_detection"] = {
            "min_pattern_size": 1,
            "max_pattern_size": 64,
            "min_count": 1_000_000,
        }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )

    first_token_at: float | None = None
    completion_tokens = 0
    accepted_prediction_tokens = 0
    rejected_prediction_tokens = 0
    output_chars = 0
    output_parts: list[str] = []
    visible_event_times: list[float] = []
    finish_reason: str | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage") or {}
                completion_tokens = max(completion_tokens, int(usage.get("completion_tokens") or 0))
                details = usage.get("completion_tokens_details") or {}
                accepted_prediction_tokens = max(
                    accepted_prediction_tokens,
                    int(details.get("accepted_prediction_tokens") or 0),
                )
                rejected_prediction_tokens = max(
                    rejected_prediction_tokens,
                    int(details.get("rejected_prediction_tokens") or 0),
                )
                choices = event.get("choices") or []
                if not choices:
                    continue
                finish_reason = choices[0].get("finish_reason") or finish_reason
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                visible = content + reasoning
                if visible:
                    now = time.monotonic()
                    first_token_at = first_token_at or now
                    visible_event_times.append(now)
                    output_chars += len(visible)
                    output_parts.append(visible)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.monotonic()
    output_text = "".join(output_parts)
    isolation_enabled = args.prompt_style == "isolation"
    own_marker = isolation_marker(request_id)
    foreign_markers = (
        [
            candidate
            for candidate in range(args.concurrency)
            if candidate != request_id and isolation_marker(candidate) in output_text
        ]
        if isolation_enabled
        else []
    )
    gaps = [
        right - left
        for left, right in zip(visible_event_times, visible_event_times[1:])
    ]
    return Result(
        request_id=request_id,
        scheduled_s=scheduled_s,
        started_s=started - epoch,
        ttft_s=None if first_token_at is None else first_token_at - started,
        elapsed_s=ended - started,
        completion_tokens=completion_tokens,
        accepted_prediction_tokens=accepted_prediction_tokens,
        rejected_prediction_tokens=rejected_prediction_tokens,
        output_chars=output_chars,
        output_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
        output_preview=output_text[:160],
        own_request_marker=not isolation_enabled or own_marker in output_text,
        foreign_request_markers=foreign_markers,
        visible_events=len(visible_event_times),
        inter_token_gap_p95_s=percentile(gaps, 0.95),
        max_inter_token_gap_s=max(gaps) if gaps else None,
        finish_reason=finish_reason,
        error=error,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--stagger-ms", type=int, default=500)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument(
        "--prompt-style",
        choices=("isolation", "mia-structured", "mia-prose"),
        default="isolation",
        help="use the isolation fixture or MiaAI-Lab's published decode prompts",
    )
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=0,
        help="ignore EOS until this many tokens; useful for decode/prefill overlap",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable GLM reasoning; direct-answer mode is the benchmark default",
    )
    parser.add_argument(
        "--disable-loop-watchdog",
        action="store_true",
        help="disable Atlas repetition detection for intentionally repetitive benchmark prompts",
    )
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args()
    if min(args.concurrency, args.prompt_tokens, args.output_tokens) <= 0 or args.stagger_ms < 0:
        parser.error("concurrency and token counts must be positive; stagger must be non-negative")
    if not 0 <= args.min_output_tokens <= args.output_tokens:
        parser.error("min-output-tokens must be between 0 and output-tokens")

    epoch = time.monotonic() + 0.25
    barrier = threading.Barrier(args.concurrency)

    def run(request_id: int) -> Result:
        barrier.wait()
        return stream_one(request_id, request_id * args.stagger_ms / 1000.0, epoch, args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(run, range(args.concurrency)))

    ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
    rates = [r.decode_tok_s for r in results if r.decode_tok_s is not None]
    first_visible = [r.started_s + r.ttft_s for r in results if r.ttft_s is not None]
    end_times = [r.started_s + r.elapsed_s for r in results]
    decode_tokens = sum(max(r.completion_tokens - 1, 0) for r in results)
    decode_window_s = max(end_times) - min(first_visible) if first_visible else None
    aggregate_window_rate = (
        decode_tokens / decode_window_s if decode_window_s and decode_window_s > 0 else None
    )
    summary = {
        "config": {
            "concurrency": args.concurrency,
            "stagger_ms": args.stagger_ms,
            "approximate_prompt_tokens": args.prompt_tokens,
            "prompt_style": args.prompt_style,
            "max_output_tokens": args.output_tokens,
            "min_output_tokens": args.min_output_tokens,
            "enable_thinking": args.enable_thinking,
            "disable_loop_watchdog": args.disable_loop_watchdog,
        },
        "requests": [{**asdict(r), "decode_tok_s": r.decode_tok_s} for r in results],
        "summary": {
            "successful": sum(r.error is None and r.ttft_s is not None for r in results),
            "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
            "ttft_p95_s": percentile(ttfts, 0.95),
            "decode_tok_s_p50": statistics.median(rates) if rates else None,
            "decode_tok_s_sum": sum(rates) if rates else None,
            "aggregate_decode_tok_s": aggregate_window_rate,
            "wall_s": time.monotonic() - epoch,
        },
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["summary"]["successful"] == args.concurrency else 2


if __name__ == "__main__":
    sys.exit(main())
