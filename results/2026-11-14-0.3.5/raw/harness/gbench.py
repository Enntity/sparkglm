#!/usr/bin/env python3
"""gbench.py [--streams N] [--ctx TOKENS] [--mode json|text] [--effort off|low] [--out N] [--salt S]
N concurrent chat requests shaped like an agent runtime's calls: a long notes context, then either a
JSON-schema answer (grammar-constrained) or free text. Prints per-stream decode tok/s and the aggregate."""
import argparse, json, threading, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--streams", type=int, default=2); ap.add_argument("--ctx", type=int, default=2000)
ap.add_argument("--mode", default="json"); ap.add_argument("--effort", default="off"); ap.add_argument("--out", type=int, default=500)
ap.add_argument("--salt", default="a"); ap.add_argument("--url", default="http://127.0.0.1:8893"); ap.add_argument("--model", default="glm-5.3-flash-atlas"); ap.add_argument("--key", default="sk-local-llm"); ap.add_argument("--warm", action="store_true"); ap.add_argument("--warm-n", type=int, default=-1); ap.add_argument("--stagger", type=float, default=0.0)
a = ap.parse_args()
KEY, MODEL = a.key, a.model
WORDS = ("harbor lantern granite willow ledger compass orchard signal copper meadow archive tunnel velvet ember quartz "
         "bridge candle furnace garden hollow island jacket kettle lumber magnet needle outpost pebble quarry ribbon").split()
def notes(n_tokens, seed):
    # roughly 1.3 tokens a word; distinct per stream so nothing is shared but the system line
    out, i, x = [], 0, seed * 7919 + 13
    while len(out) * 1.35 < n_tokens:
        x = (x * 1103515245 + 12345) % 2**31
        out.append(WORDS[x % len(WORDS)]); i += 1
        if i % 17 == 0: out.append(f"\nEntry {i // 17} ({seed}):")
    return " ".join(out)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "minItems": 12, "maxItems": 12,
          "items": {"type": "object", "additionalProperties": False, "required": ["entry", "summary", "mood", "weight"],
                    "properties": {"entry": {"type": "integer"}, "summary": {"type": "string"}, "mood": {"type": "string", "enum": ["calm", "tense", "bright", "flat"]},
                                   "weight": {"type": "number"}}}}}}
def body(i):
    task = ("Pick twelve entries from the notes and record an appraisal of each: a one-sentence summary of at least fifteen words, a mood and a weight."
            if a.mode == "json" else "Write a detailed twelve-paragraph reflection on the notes above, quoting entry numbers.")
    b = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": a.out,
         "messages": [{"role": "system", "content": "You are a careful archivist."},
                      {"role": "user", "content": f"Notes ({a.salt}-{i}):\n{notes(a.ctx, hash((a.salt, i)) % 100000)}\n\n{task}"}]}
    if a.mode == "json": b["response_format"] = {"type": "json_schema", "json_schema": {"name": "appraisals", "strict": True, "schema": SCHEMA}}
    if a.effort == "off": b.update(reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
    else: b["reasoning_effort"] = a.effort
    return b
def run(i, res, b):
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json", "authorization": "Bearer " + KEY})
    t0 = time.time(); t1 = tl = None; usage = {}; gaps = []
    with urllib.request.urlopen(req, timeout=1800) as h:
        for line in h:
            if not line.startswith(b"data:"): continue
            d = line[5:].strip()
            if d == b"[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            ch = (j.get("choices") or [{}])[0].get("delta") or {}
            if ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning"):
                now = time.time()
                if tl: gaps.append(now - tl)
                t1 = t1 or now; tl = now
    if t1 is None: t1 = tl = time.time()
    ct = usage.get("completion_tokens", 0)
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    res[i] = dict(prompt=usage.get("prompt_tokens"), tokens=ct, ttft=t1 - t0, decode_s=tl - t1, tps=(ct - 1) / (tl - t1) if tl > t1 else 0, t_first=t1, t_last=tl,
                  cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"), reasoning=rt, max_gap=max(gaps or [0]), stalled=sum(g for g in gaps if g > 0.5))
bodies = [body(i) for i in range(a.streams)]
if a.warm:  # prefill each context first, one at a time, so the measured wave decodes with no prefill beside it
    for b in (bodies if a.warm_n < 0 else bodies[:a.warm_n]):
        w = dict(b, max_tokens=1); r = [None]; run(0, r, w)
res = [None] * a.streams
th = [threading.Thread(target=run, args=(i, res, bodies[i])) for i in range(a.streams)]
t0 = time.time()
for t in th: t.start(); time.sleep(a.stagger)
[t.join() for t in th]
first = min(r["t_first"] for r in res); last = max(r["t_last"] for r in res); wall = last - t0
agg = sum(r["tokens"] for r in res) / (last - first)
print(json.dumps({"case": f"c{a.streams}-{a.mode}-{a.effort}-ctx{a.ctx}{'-warm' if a.warm else ''}", "e2e_tok_s": round(sum(r["tokens"] for r in res) / wall, 1), "wall_s": round(wall, 1), "aggregate_decode_tok_s": round(agg, 1),
                  "per_stream_tok_s": [round(r["tps"], 1) for r in res], "tokens": [r["tokens"] for r in res], "reasoning": [r["reasoning"] for r in res], "prompt": [r["prompt"] for r in res],
                  "cached": [r["cached"] for r in res], "ttft_s": [round(r["ttft"], 2) for r in res], "max_gap_s": [round(r["max_gap"], 2) for r in res], "stalled_s": [round(r["stalled"], 1) for r in res]}))
