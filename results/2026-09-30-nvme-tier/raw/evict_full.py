#!/usr/bin/env python3
"""evict_full.py TAG [CONVS=10] [RECORDS=8000]: overflow the full KV pool with long conversations, then return to the
earliest ones. Turn 1 of every conversation is cold and sequential (~190K tokens each at 8000 records); turn 2 is
asked of conversations 0-2 (evicted from the GPU pool by then) and of the last one (still resident)."""
import json, random, sys, time, urllib.request
URL = "http://127.0.0.1:8893/v1/chat/completions"
tag = sys.argv[1]; C = int(sys.argv[2]) if len(sys.argv) > 2 else 10; R = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
def doc(c):
    rng = random.Random(1000 + c)
    rows = [(i, rng.randint(100000, 999999), rng.randint(1, 99)) for i in range(R)]
    text = f"Ledger {c}-{rng.randint(1000,9999)}\n" + "\n".join(f"Record {i}: unit-{c:02d}-{i:05d} has access code {k} and owner team-{t}." for i, k, t in rows)
    return text, rows
def chat(msgs):
    body = {"model": "glm-5.3-flash-atlas", "max_tokens": 64, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}
    t = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                p = (ch.get("delta") or {}).get("content") or ""
                if p and ttft is None: ttft = time.time() - t
                text += p
    return text, round(ttft or -1, 2), usage.get("prompt_tokens"), (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
hist, out = {}, []
for c in range(C):
    text, rows = doc(c); i = (37 * c + 11) % R
    msgs = [{"role": "user", "content": text + f"\n\nWhat is the access code of unit-{c:02d}-{i:05d}? Answer with the number only."}]
    a, ttft, n, cached = chat(msgs); good = str(rows[i][1]) in a
    hist[c] = (msgs + [{"role": "assistant", "content": a}], rows)
    out.append({"conv": c, "turn": 1, "ttft": ttft, "prompt": n, "cached": cached, "correct": good}); print(out[-1], flush=True)
for c in (0, 1, 2, C - 1):
    msgs, rows = hist[c]; i = (53 * c + 29) % R
    a, ttft, n, cached = chat(msgs + [{"role": "user", "content": f"And the access code of unit-{c:02d}-{i:05d}? Number only."}])
    out.append({"conv": c, "turn": 2, "ttft": ttft, "prompt": n, "cached": cached, "correct": str(rows[i][1]) in a}); print(out[-1], flush=True)
json.dump(out, open(f"evict-{tag}.json", "w"), indent=1)
print(f"{tag}: {sum(o['correct'] for o in out)}/{len(out)} correct; turn-2 TTFT", [o["ttft"] for o in out if o["turn"] == 2])
