#!/usr/bin/env python3
"""soak.py TAG [MINUTES=20]: concurrent image + text traffic with checkable answers.

Four workers for MINUTES, greedy, thinking off:
  img-a, img-b  three-turn image conversations: a random shape (red circle, blue
                square, green triangle) behind 2K-30K tokens of code context;
                every turn's answer must name the image's colour
  text          needle retrieval from a ~60K-token record document, three
                follow-ups per document; each answer must be the exact code
  short         small arithmetic questions; the answer must be exact
Writes soak-TAG.json: per-worker requests, failures (wrong answer or error),
TTFT percentiles. Needs vpc_gate.py and code.txt beside it.
"""
import json, os, random, sys, threading, time

import vpc_gate as g

TAG = sys.argv[1]
MINUTES = float(sys.argv[2]) if len(sys.argv) > 2 else 20
END = time.time() + MINUTES * 60
CODE = open(os.path.join(g.HERE, "code.txt")).read()
SHAPES = [(g.P, "red"), (g.Q, "blue"), (g.R, "green")]
stats = {}
lock = threading.Lock()


def record(worker, ok, r=None, err=None, note=None):
    with lock:
        s = stats.setdefault(worker, {"requests": 0, "failures": 0, "ttft": [], "examples": []})
        s["requests"] += 1
        if r:
            s["ttft"].append(r["ttft"])
        if not ok:
            s["failures"] += 1
            if len(s["examples"]) < 5:
                s["examples"].append({"error": err, "text": (r or {}).get("text"), "note": note})


def call(worker, msgs, check, note, max_tokens=40):
    try:
        r = g.stream(msgs, max_tokens=max_tokens)
        ok = check(r["text"])
        record(worker, ok, r, note=None if ok else note)
        return r
    except Exception as e:  # noqa: BLE001 - every failure counts
        record(worker, False, err=repr(e)[:300], note=note)
        return None


def image_worker(name, seed):
    rng = random.Random(seed)
    while time.time() < END:
        img, colour = rng.choice(SHAPES)
        start = rng.randrange(0, len(CODE) - 120000)
        ctx = CODE[start:start + rng.randrange(8000, 120000)]
        msgs = [{"role": "system", "content": "Answer briefly.\n\nContext:\n" + ctx},
                {"role": "user", "content": [img, g.text("What colour is the shape in this image? One word.")]}]
        for q in ["What colour was it? One word.", "Name the colour once more. One word.", None]:
            r = call(name, msgs, lambda t, c=colour: c in t.lower(), f"expected {colour}")
            if r is None or q is None or time.time() >= END:
                break
            msgs += [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": q}]


def text_worker(seed):
    rng = random.Random(seed)
    while time.time() < END:
        codes = [rng.randint(100000, 999999) for _ in range(2400)]
        doc = "\n".join(f"Record {i}: unit-{seed}-{i:05d} has access code {c}." for i, c in enumerate(codes))
        msgs = [{"role": "system", "content": "Answer exactly.\n\n" + doc}]
        for _ in range(3):
            i = rng.randrange(len(codes))
            msgs.append({"role": "user", "content": f"What is the access code of unit-{seed}-{i:05d}? Number only."})
            r = call("text", msgs, lambda t, c=str(codes[i]): c in t, f"expected {codes[i]}", max_tokens=16)
            if r is None or time.time() >= END:
                break
            msgs.append({"role": "assistant", "content": r["text"]})
        seed += 1


def short_worker(seed):
    rng = random.Random(seed)
    while time.time() < END:
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        msgs = [{"role": "user", "content": f"What is {a} times {b}? Number only."}]
        call("short", msgs, lambda t, v=str(a * b): v in t.replace(",", ""), f"expected {a * b}", max_tokens=12)


threads = [threading.Thread(target=image_worker, args=("img-a", 1)),
           threading.Thread(target=image_worker, args=("img-b", 2)),
           threading.Thread(target=text_worker, args=(700,)),
           threading.Thread(target=short_worker, args=(9,))]
t0 = time.time()
[t.start() for t in threads]
[t.join() for t in threads]
out = {"tag": TAG, "minutes": round((time.time() - t0) / 60, 1), "workers": {}}
for w, s in sorted(stats.items()):
    tt = sorted(s["ttft"])
    pct = lambda p: tt[min(len(tt) - 1, int(p * len(tt)))] if tt else None
    out["workers"][w] = {"requests": s["requests"], "failures": s["failures"],
                         "ttft_p50": pct(0.5), "ttft_p95": pct(0.95), "examples": s["examples"]}
out["requests"] = sum(s["requests"] for s in stats.values())
out["failures"] = sum(s["failures"] for s in stats.values())
json.dump(out, open(f"soak-{TAG}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "workers"}),
      {w: (v["requests"], v["failures"], v["ttft_p50"], v["ttft_p95"]) for w, v in out["workers"].items()}, flush=True)
