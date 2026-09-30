#!/usr/bin/env python3
"""Capacity check: four concurrent long sessions, then a cached follow-up turn each.

Usage: pc_fill.py TAG [RECORDS]   (default 9600 records, about 240K tokens per session)
Exact-answer retrieval from early, middle and late in each document.
"""
import json, random, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8893/v1/chat/completions"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 9600


def call(messages):
    body = {"model": "glm-5.3-flash-atlas", "messages": messages, "max_tokens": 24, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return text.strip(), round(ttft or -1, 2), usage.get("prompt_tokens"), (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


results = []
def session(s):
    rng = random.Random(500 + s)
    codes = [rng.randint(100000, 999999) for _ in range(N)]
    rows = [f"Record {i}: unit-{s}-{i:05d} has access code {c}." for i, c in enumerate(codes)]
    convo = [{"role": "system", "content": "Answer exactly.\n\n" + "\n".join(rows)}]
    for turn, i in enumerate((N // 10, N // 2, N - N // 10), 1):
        convo.append({"role": "user", "content": f"What is the access code of unit-{s}-{i:05d}? Number only."})
        text, ttft, prompt, cached = call(convo)
        results.append({"session": s, "turn": turn, "ttft": ttft, "prompt": prompt, "cached": cached,
                        "correct": str(codes[i]) in text, "text": text})
        print("session", s, "turn", turn, ttft, prompt, cached, str(codes[i]) in text, flush=True)
        convo.append({"role": "assistant", "content": text})

t0 = time.time()
threads = [threading.Thread(target=session, args=(s,)) for s in range(4)]
[t.start() for t in threads]; [t.join() for t in threads]
summary = {"tag": sys.argv[1], "correct": sum(r["correct"] for r in results), "n": len(results),
           "prompt_tokens_max": max(r["prompt"] or 0 for r in results), "wall_s": round(time.time() - t0, 1)}
json.dump({"summary": summary, "results": results}, open(f"fill-{sys.argv[1]}.json", "w"), indent=1)
print(json.dumps(summary))
