#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Minimal concurrent-sequence isolation probe for an OpenAI endpoint.

Each request carries a different nonce after a long, otherwise identical
prompt.  A correct server must return only that request's nonce; returning a
neighbour's nonce is direct evidence of slot/logits/state cross-contamination.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request


NONCES = ("ALPHA53", "BRAVO17", "CHARLIE29", "DELTA41")


def prompt(nonce: str, approximate_tokens: int) -> str:
    filler = " ".join(f"neutral{i}" for i in range(approximate_tokens))
    return (
        f"Ignore this neutral padding: {filler}\n"
        f"The answer for this request is {nonce}. Return exactly {nonce} and nothing else."
    )


def request_one(base_url: str, model: str, index: int, tokens: int, stagger_ms: int) -> dict:
    time.sleep(index * stagger_ms / 1000)
    nonce = NONCES[index]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt(nonce, tokens)}],
        "temperature": 0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    output = (message.get("reasoning_content") or "") + (message.get("content") or "")
    foreign = [candidate for candidate in NONCES if candidate != nonce and candidate in output]
    return {
        "request": index,
        "expected": nonce,
        "output": output,
        "own_nonce": nonce in output,
        "foreign_nonces": foreign,
        "elapsed_s": time.monotonic() - started,
        "ttft_ms": result.get("time_to_first_token_ms"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--stagger-ms", type=int, default=500)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(NONCES)) as pool:
        rows = list(
            pool.map(
                lambda i: request_one(
                    args.base_url, args.model, i, args.prompt_tokens, args.stagger_ms
                ),
                range(len(NONCES)),
            )
        )
    passed = all(row["own_nonce"] and not row["foreign_nonces"] for row in rows)
    print(json.dumps({"passed": passed, "requests": rows}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
