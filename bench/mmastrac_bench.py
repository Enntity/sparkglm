#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Replicates mmastrac/glm-5.3-flash-4x-gx10 dev/repro prefill.py + decode.py against Atlas."""
import json, statistics, sys, time, urllib.request, uuid
BASE = "http://127.0.0.1:8893/v1"
MODEL = "glm-5.3-flash-atlas"

def stream(body):
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if ttft is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                    ttft = time.time() - t0
    return ttft, time.time() - t0, usage

PER_UNIT = None

def prefill(target, units=None):
    nonce = uuid.uuid4().hex
    unit = "Reading {} note {}: the pump was serviced and the filter replaced.\n"
    units = units if units is not None else round(target / PER_UNIT)
    text = "".join(unit.format(nonce, n) for n in range(units)) + "\nReply with OK."
    ttft, _, u = stream({"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 4,
                         "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                         "chat_template_kwargs": {"thinking": False, "enable_thinking": False}})
    return u["prompt_tokens"], ttft

PROMPTS = {"structured": "Count from 1 to 200, comma separated. No commentary.",
           "code": "Write a red-black tree in Python with insert, delete and rebalancing. Code only.",
           "prose": "Explain how a hash map works, in flowing prose. No code, no lists."}

def decode(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 512, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    _, el, u = stream(body)
    return u["completion_tokens"] / el, u["completion_tokens"], (u.get("completion_tokens_details") or {}).get("accepted_prediction_tokens")

# Calibrate tokens per unit line with a warm-up probe (the lines are ~40 tokens).
PER_UNIT = prefill(0, units=100)[0] / 100
for target in [int(x) for x in sys.argv[1:]] or [32000, 128000]:
    prefill(2000)  # warm
    n, t = prefill(target)
    print(f"prefill {n} tokens: TTFT {t:.2f}s = {n/t:.0f} tok/s", flush=True)
for name, p in PROMPTS.items():
    decode(p)
    runs = [decode(p) for _ in range(3)]
    med = statistics.median(r[0] for r in runs)
    print(f"decode {name}: {med:.1f} tok/s (runs {[round(r[0],1) for r in runs]}, tokens {[r[1] for r in runs]}, accepted {[r[2] for r in runs]})", flush=True)
