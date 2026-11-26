#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Quality probe: quality_probe.py TAG [CONC].
Deterministic, exactly checkable items at temperature 0, thinking off:
  * 40 four-digit multi-step arithmetic problems (answer = one integer);
  * 12 two-hop needle retrievals over ~24K-token key/code/value tables.
Prints per-set accuracy and writes ./qprobe-TAG.jsonl (one line per item)."""
import json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

TAG = sys.argv[1]
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
URL = "http://127.0.0.1:8893/v1/chat/completions"


def ask(prompt, max_tokens):
    body = {"model": "glm-5.3-flash-atlas", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    msg = json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]
    return msg.get("content") or ""


def arithmetic(rng):
    a, b, c, d, e = (rng.randint(1000, 9999), rng.randint(13, 97), rng.randint(1000, 9999),
                     rng.randint(13, 97), rng.randint(100, 999))
    q = (f"Compute {a} * {b} - {c} * {d} + {e} exactly, showing brief steps. "
         "Give the final answer on the last line as 'Answer: <number>'.")
    return q, str(a * b - c * d + e)


def needle(rng):
    # Two hops over ~24K tokens: key -> code (table A), code -> value (table B).
    keys = [f"k{rng.randrange(10**6):06d}" for _ in range(900)]
    codes = [f"c{rng.randrange(10**6):06d}" for _ in keys]
    vals = [f"{rng.randrange(10**8):08d}" for _ in keys]
    i = rng.randrange(len(keys))
    a = "\n".join(f"{k} -> {c}" for k, c in zip(keys, codes))
    order = list(range(len(keys))); rng.shuffle(order)
    b = "\n".join(f"{codes[j]} = {vals[j]}" for j in order)
    q = (f"Table A maps keys to codes:\n{a}\n\nTable B maps codes to values:\n{b}\n\n"
         f"Find the code for key {keys[i]} in Table A, then that code's value in Table B. "
         "Reply with the 8-digit value only.")
    return q, vals[i]


def grade(kind, out, answer):
    if kind == "arith":
        m = re.findall(r"Answer:\s*\$?(-?[\d,]+)", out)
        return bool(m) and m[-1].replace(",", "") == answer
    return answer in out


rng = random.Random(20260928)
items = [("arith", *arithmetic(rng)) for _ in range(40)] + [("needle", *needle(rng)) for _ in range(12)]


def run(item):
    kind, q, answer = item
    out = ask(q, 400 if kind == "arith" else 24)
    return {"tag": TAG, "kind": kind, "ok": grade(kind, out, answer), "answer": answer, "out": out[-120:]}


with ThreadPoolExecutor(CONC) as ex:
    results = list(ex.map(run, items))
with open(f"qprobe-{TAG}.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
for kind in ("arith", "needle"):
    rs = [r for r in results if r["kind"] == kind]
    print(f"{TAG} {kind}: {sum(r['ok'] for r in rs)}/{len(rs)}")
