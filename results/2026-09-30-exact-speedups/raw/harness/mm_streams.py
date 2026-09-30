#!/usr/bin/env python3
"""Aggregate decode throughput at 1/2/4/8 concurrent streams (mmastrac's three decode prompts, cycled; 512 tokens each)."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
B = "http://127.0.0.1:8893/v1/chat/completions"
P = ["<prompt omitted: quoted from mmastrac/glm-5.3-flash-4x-gx10 @ 4e63b64, which has no license>",
     "<prompt omitted: quoted from mmastrac/glm-5.3-flash-4x-gx10 @ 4e63b64, which has no license>",
     "<prompt omitted: quoted from mmastrac/glm-5.3-flash-4x-gx10 @ 4e63b64, which has no license>"]
def one(p):
    b = {"model": "glm-5.3-flash-atlas", "messages": [{"role": "user", "content": p}], "max_tokens": 512,
         "temperature": 0, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(B, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
    return r["usage"]["completion_tokens"]
one(P[2])
for n in (1, 2, 4, 8):
    t = time.time()
    with ThreadPoolExecutor(n) as ex:
        toks = sum(ex.map(one, [P[i % 3] for i in range(n)]))
    el = time.time() - t
    print(f"streams {n}: {toks} tokens in {el:.1f}s = {toks/el:.1f} tok/s aggregate", flush=True)
