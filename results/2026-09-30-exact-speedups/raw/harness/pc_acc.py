#!/usr/bin/env python3
"""Accuracy parity, prefix cache on vs off: 4 documents x 10 turns = 40 questions.

Usage: pc_acc.py TAG   Greedy, thinking off, one session at a time. Reports exact
answers, off-by-one-record slips and the median time to first token after turn 1.
"""
import json, random, statistics, sys, time, urllib.request

URL = "http://127.0.0.1:8893/v1/chat/completions"


def call(messages):
    body = {"model": "glm-5.3-flash-atlas", "messages": messages, "max_tokens": 24, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; cached = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                cached = (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return text.strip(), ttft or -1, cached


results = []
for doc in range(4):
    rng = random.Random(1000 + doc)
    names = [f"unit-{doc}-{i:05d}" for i in range(1300 + 200 * doc)]
    codes = [rng.randint(100000, 999999) for _ in names]
    rows = [f"Record {i}: {n} has access code {c} and owner team-{rng.randint(1, 99)}." for i, (n, c) in enumerate(zip(names, codes))]
    convo = [{"role": "system", "content": "Answer questions about these records exactly.\n\n" + "\n".join(rows)}]
    for turn, i in enumerate(random.Random(doc).sample(range(len(names)), 10), 1):
        convo.append({"role": "user", "content": f"What is the access code of {names[i]}? Answer with the number only."})
        text, ttft, cached = call(convo)
        near = {codes[j] for j in (i - 1, i + 1) if 0 <= j < len(codes)}
        results.append({"doc": doc, "turn": turn, "correct": str(codes[i]) in text,
                        "off_by_one": any(str(c) in text for c in near), "ttft": ttft, "cached": cached, "text": text})
        convo.append({"role": "assistant", "content": text})
        print(doc, turn, results[-1]["correct"], results[-1]["off_by_one"], round(ttft, 2), cached, flush=True)

warm = [r["ttft"] for r in results if r["turn"] > 1]
summary = {"tag": sys.argv[1], "correct": sum(r["correct"] for r in results), "off_by_one": sum(r["off_by_one"] for r in results),
           "n": len(results), "median_ttft_turns_2_10": round(statistics.median(warm), 2)}
json.dump({"summary": summary, "results": results}, open(f"acc-{sys.argv[1]}.json", "w"), indent=1)
print(json.dumps(summary))
