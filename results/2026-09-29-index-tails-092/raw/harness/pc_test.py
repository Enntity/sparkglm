#!/usr/bin/env python3
"""Prefix-cache check: replay and an omp-style multi-turn conversation.

Usage: pc_test.py TAG   (writes pc-TAG.json)
Greedy, thinking off. Every answer is exactly checkable, so a cache hit that
restores the wrong state shows up as a wrong answer, not just a slow one.
"""
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:8893/v1/chat/completions"
MODEL = "glm-5.3-flash-atlas"
TAG = sys.argv[1]


def stream(messages, max_tokens=64):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return {"ttft": round(ttft or -1, 2), "wall": round(time.time() - t0, 2), "text": text.strip(),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}


def document(n_records, seed):
    rng = random.Random(seed)
    rows, facts = [], {}
    for i in range(n_records):
        name = f"unit-{seed}-{i:05d}"
        code = rng.randint(100000, 999999)
        rows.append(f"Record {i}: {name} has access code {code} and owner team-{rng.randint(1, 99)}.")
        facts[name] = code
    return "\n".join(rows), facts


results = {"tag": TAG, "replay": [], "turns": []}

# Replay: the same ~32K-token prompt twice.
doc, facts = document(1400, 7)
key = sorted(facts)[700]
msgs = [{"role": "user", "content": doc + f"\n\nWhat is the access code of {key}? Answer with the number only."}]
for attempt in (1, 2):
    r = stream(msgs)
    r["correct"] = str(facts[key]) in r["text"]
    results["replay"].append(r)
    print("replay", attempt, {k: r[k] for k in ("ttft", "prompt_tokens", "cached", "correct")}, repr(r["text"][:40]), flush=True)

# Multi-turn: a ~40K-token document, then six turns that each append the answer
# and a new question, the way an agent re-sends its whole conversation.
doc, facts = document(1800, 11)
keys = random.Random(3).sample(sorted(facts), 6)
convo = [{"role": "system", "content": "You answer questions about the records below exactly.\n\n" + doc}]
for turn, key in enumerate(keys, 1):
    convo.append({"role": "user", "content": f"What is the access code of {key}? Answer with the number only."})
    r = stream(convo)
    r["correct"] = str(facts[key]) in r["text"]
    r["turn"] = turn
    results["turns"].append(r)
    print("turn", turn, {k: r[k] for k in ("ttft", "prompt_tokens", "cached", "correct")}, repr(r["text"][:40]), flush=True)
    convo.append({"role": "assistant", "content": r["text"]})

json.dump(results, open(f"pc-{TAG}.json", "w"), indent=1)
ok = sum(r["correct"] for r in results["replay"] + results["turns"])
print(f"{TAG}: {ok}/{len(results['replay']) + len(results['turns'])} correct")
