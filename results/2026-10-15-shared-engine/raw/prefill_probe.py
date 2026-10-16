#!/usr/bin/env python3
"""prefill_probe.py TAG [SIZES...]: cold-prefill TTFT of unique prompts (no cache hits), greedy, thinking off,
one output token. Two reps per size. Writes prefill-TAG.json in the cwd."""
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:8893"
MODEL = "glm-5.3-flash-atlas"
TAG = sys.argv[1]
SIZES = [int(s) for s in sys.argv[2:]] or [8192, 32768, 65536, 131072]


def make(n, salt):
    # ~69 tokens per line of 12 short words + header; usage reports the exact count
    rng = random.Random(salt)
    words = [f"w{rng.randint(0, 99999)}" for _ in range(60)]
    lines = [f"R{salt}-{i}: " + " ".join(rng.choice(words) for _ in range(12)) + "." for i in range(int(n / 69))]
    return "\n".join(lines)


out = []
for n in SIZES:
    for rep in range(2):
        text = make(n, f"{rep}{time.time_ns() % 100000}")
        body = {"model": MODEL, "messages": [{"role": "user", "content": text + "\n\nReply with OK."}],
                "max_tokens": 1, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        ttft = time.time() - t0; have = r["usage"]["prompt_tokens"]
        out.append({"size": n, "rep": rep, "tokens": have, "ttft": round(ttft, 3)})
        print(out[-1], flush=True)
json.dump({"tag": TAG, "results": out}, open(f"prefill-{TAG}.json", "w"), indent=1)
