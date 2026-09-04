#!/usr/bin/env python3
"""Fast streamed concurrency smoke test for the synthetic tinyGLM endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request


def prompt_for_tokens(count: int, stream: int, prompt_seed: int = 0) -> str:
    # All tokens are in tinyGLM's WordLevel vocabulary and whitespace maps one
    # word to one token. The per-stream rotation avoids identical-prefix hits.
    return " ".join(
        f"t{(index + stream * 17 + prompt_seed * 31) % 200}"
        for index in range(count)
    )


def run_stream(
    endpoint: str,
    model: str,
    prompt_tokens: int,
    output_tokens: int,
    stream: int,
    stagger_ms: int,
    timeout: float,
    prompt_seed: int = 0,
) -> dict:
    time.sleep(stream * stagger_ms / 1000.0)
    sent = time.perf_counter()
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt_for_tokens(prompt_tokens, stream, prompt_seed),
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first = None
    completion_tokens = 0
    chunks = 0
    token_ids: list[int] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            if choices:
                chunks += 1
                if first is None:
                    first = time.perf_counter()
                token_ids.extend(int(token) for token in choices[0].get("token_ids") or [])
            usage = event.get("usage") or {}
            completion_tokens = max(
                completion_tokens, int(usage.get("completion_tokens") or 0)
            )
    ended = time.perf_counter()
    if first is None:
        first = ended
    if completion_tokens == 0:
        completion_tokens = len(token_ids) or chunks
    return {
        "stream": stream,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": first - sent,
        "wall_s": ended - sent,
        "decode_tok_s": completion_tokens / max(ended - first, 1e-9),
        "token_ids_count": len(token_ids),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="tinyGLM-5.3-EXL3")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--stagger-ms", type=int, default=250)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def benchmark(
    *,
    endpoint: str,
    model: str,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
    stagger_ms: int,
    timeout: float,
    prompt_seed: int = 0,
) -> dict:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [
            executor.submit(
                run_stream,
                endpoint,
                model,
                prompt_tokens,
                output_tokens,
                stream,
                stagger_ms,
                timeout,
                prompt_seed,
            )
            for stream in range(concurrency)
        ]
        streams = [future.result() for future in futures]
    wall = time.perf_counter() - started
    total = sum(item["completion_tokens"] for item in streams)
    return {
        "model": model,
        "concurrency": concurrency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "stagger_ms": stagger_ms,
        "wall_s": wall,
        "aggregate_tok_s": total / wall,
        "streams": sorted(streams, key=lambda item: item["stream"]),
    }


def main() -> None:
    args = parse_args()
    result = benchmark(
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        stagger_ms=args.stagger_ms,
        timeout=args.timeout,
        prompt_seed=0,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(
        f"tinyGLM C{args.concurrency} stagger={args.stagger_ms}ms "
        f"aggregate={result['aggregate_tok_s']:.1f} tok/s "
        f"wall={result['wall_s']:.3f}s"
    )
    for item in result["streams"]:
        print(
            f"  s{item['stream']}: TTFT={item['ttft_s']:.3f}s "
            f"decode={item['decode_tok_s']:.1f} tok/s "
            f"tokens={item['completion_tokens']} wall={item['wall_s']:.3f}s"
        )


if __name__ == "__main__":
    main()
