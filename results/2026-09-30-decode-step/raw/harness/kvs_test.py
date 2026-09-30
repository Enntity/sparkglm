#!/usr/bin/env python3
"""Greedy equality probe for KV sharding: fixed prompts, record text, TTFT and decode rate. Usage: kvs_test.py TAG"""
import json, random, sys, time, urllib.request
URL = "http://127.0.0.1:8893/v1/chat/completions"
def call(content, max_tokens):
    body = {"model": "glm-5.3-flash-atlas", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); usage = d.get("usage") or usage
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None: ttft = time.time() - t0
                text += piece
    wall = time.time() - t0; n = usage.get("completion_tokens") or 0
    return {"text": text, "ttft": round(ttft or -1, 2), "decode_tok_s": round(n / max(wall - (ttft or 0), 1e-6), 1), "prompt": usage.get("prompt_tokens")}
def doc(n, seed):
    rng = random.Random(seed); codes = [rng.randint(100000, 999999) for _ in range(n)]
    return "\n".join(f"Record {i}: unit-{i:05d} has access code {c} and owner team-{rng.randint(1,99)}." for i, c in enumerate(codes)), codes
cases = {"short": ("Write a Python function that merges two sorted lists, with a docstring and two doctests.", 256)}
for name, n in (("16k", 640), ("64k", 2600), ("128k", 5200)):
    d, codes = doc(n, 900 + n)
    k = n * 3 // 7
    cases[name] = (d + f"\n\nWhat is the access code of unit-{k:05d}? Then list the owner teams of records 0 to 9, one per line.", 160)
    cases[name + "_answer"] = str(codes[k])
out = {}
for name in ("short", "16k", "64k", "128k"):
    content, mt = cases[name]; r = call(content, mt)
    if name != "short": r["needle_correct"] = cases[name + "_answer"] in r["text"]
    out[name] = r; print(name, r["prompt"], r["ttft"], r["decode_tok_s"], r.get("needle_correct"), repr(r["text"][:60]), flush=True)
json.dump(out, open(f"kvs-{sys.argv[1]}.json", "w"), indent=1)
