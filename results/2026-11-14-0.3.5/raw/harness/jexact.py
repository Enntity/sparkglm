#!/usr/bin/env python3
"""jexact.py [--n 8] [--out 300] [--ctx 600]: greedy outputs must not depend on how many sequences share a step.
Each prompt is generated alone, then in waves of 2 and of 4 (with different partners), JSON-schema and free text.
Prints how many outputs are identical to the lone run, and the first difference if any."""
import argparse, json, threading, urllib.request, hashlib
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", type=int, default=300)
ap.add_argument("--ctx", type=int, default=600); ap.add_argument("--url", default="http://127.0.0.1:8893"); ap.add_argument("--salt", default="x")
a = ap.parse_args()
import importlib.util, sys
WORDS = ("harbor lantern granite willow ledger compass orchard signal copper meadow archive tunnel velvet ember quartz "
         "bridge candle furnace garden hollow island jacket kettle lumber magnet needle outpost pebble quarry ribbon").split()
def notes(n_tokens, seed):
    out, i, x = [], 0, seed * 7919 + 13
    while len(out) * 1.35 < n_tokens:
        x = (x * 1103515245 + 12345) % 2**31; out.append(WORDS[x % len(WORDS)]); i += 1
        if i % 17 == 0: out.append(f"\nEntry {i // 17} ({seed}):")
    return " ".join(out)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "minItems": 6, "maxItems": 6,
          "items": {"type": "object", "additionalProperties": False, "required": ["entry", "summary", "mood", "weight"],
                    "properties": {"entry": {"type": "integer"}, "summary": {"type": "string"}, "mood": {"type": "string", "enum": ["calm", "tense", "bright", "flat"]},
                                   "weight": {"type": "number"}}}}}}
def body(i, mode):
    task = ("Pick six entries from the notes and record an appraisal of each: a one-sentence summary of at least fifteen words, a mood and a weight."
            if mode == "json" else "Write a six-paragraph reflection on the notes above, quoting entry numbers.")
    b = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": a.out,
         "messages": [{"role": "system", "content": "You are a careful archivist."},
                      {"role": "user", "content": f"Notes ({a.salt}-{mode}-{i}):\n{notes(a.ctx, 1000 + i)}\n\n{task}"}]}
    if mode == "json":
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "appraisals", "strict": True, "schema": SCHEMA}}
        b.update(reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
    else: b["reasoning_effort"] = "low"
    return b
def call(b):
    r = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json", "authorization": "Bearer sk-local-llm"})
    with urllib.request.urlopen(r, timeout=900) as f: d = json.load(f)
    m = d["choices"][0]["message"]; return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or ""), d["usage"]["completion_tokens"]
def wave(bodies):
    out = [None] * len(bodies)
    def run(i): out[i] = call(bodies[i])
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]; [t.start() for t in th]; [t.join() for t in th]; return out
ok_all = True
for mode in ("json", "text"):
    bodies = [body(i, mode) for i in range(a.n)]
    lone = [call(b) for b in bodies]
    again = [call(b) for b in bodies]
    same_again = sum(x[0] == y[0] for x, y in zip(lone, again))
    res = {}
    for width, order in ((2, list(range(a.n))), (4, list(range(a.n))), (4, [(i * 3) % a.n for i in range(a.n)]), (3, list(range(a.n - a.n % 3)))):
        got = {}
        for k in range(0, len(order), width):
            grp = order[k:k + width]
            for i, o in zip(grp, wave([bodies[i] for i in grp])): got[i] = o
        same = sum(got[i][0] == lone[i][0] for i in got)
        res[f"c{width}{'b' if order != list(range(len(order))) else ''}"] = f"{same}/{len(got)}"
        if same != len(got):
            ok_all = False
            i = next(i for i in got if got[i][0] != lone[i][0]); x, y = lone[i][0], got[i][0]
            p = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print(f"  DIFF {mode} c{width} prompt {i} at char {p}: lone={x[max(0,p-40):p+40]!r} wave={y[max(0,p-40):p+40]!r}")
    print(mode, "tokens", [t for _, t in lone], "| lone repeat identical", f"{same_again}/{a.n}", "| waves identical to lone:", res)
print("RESULT:", "IDENTICAL" if ok_all else "DIFFERENT")
