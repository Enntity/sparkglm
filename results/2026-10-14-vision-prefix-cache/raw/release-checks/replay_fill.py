#!/usr/bin/env python3
"""replay_fill.py TAG: replay the fill check's misses on the vision release candidate as single cold requests,
with the earlier answers each round recorded, and print the answers.
  fill2 s0 t2: FILL_SEED 9002, session 0, turn 2 (right 853374; the cached turn answered 524770, record 4840's code)
  fill4 s3 t3: FILL_SEED 9004, session 3, turn 3 (right 610460; the cached turn answered "6", and so did fill5 s2 t3,
               the same codes under unit-2 labels)"""
import json, random, sys
import vpc_gate as g
N = 9600; out = {}
for rnd, s, turn in ((2, 0, 2), (4, 3, 3)):
    rng = random.Random(9000 + rnd + s); codes = [rng.randint(100000, 999999) for _ in range(N)]
    rows = [f"Record {i}: unit-{s}-{i:05d} has access code {c}." for i, c in enumerate(codes)]
    rec = {r["turn"]: r["text"] for r in json.load(open(f"fill-rel-fill{rnd}.json"))["results"] if r["session"] == s}
    msgs = [{"role": "system", "content": "Answer exactly.\n\n" + "\n".join(rows)}]
    for t, i in enumerate((N // 10, N // 2, N - N // 10)[:turn], 1):
        msgs.append({"role": "user", "content": f"What is the access code of unit-{s}-{i:05d}? Number only."})
        if t < turn:
            msgs.append({"role": "assistant", "content": rec[t]})
    right = codes[(N // 10, N // 2, N - N // 10)[turn - 1]]
    r = g.stream(msgs, max_tokens=24)
    print(f"replay {sys.argv[1]} fill{rnd} s{s} t{turn}: answer {r['text']!r} (right {right}; cached turn said {rec[turn]!r}), "
          f"cached {r['cached']}, prompt {r['prompt_tokens']}", flush=True)
    out[f"fill{rnd}-s{s}-t{turn}"] = {"reply": r, "right": right, "cached_turn": rec[turn]}
json.dump(out, open(f"replay-{sys.argv[1]}.json", "w"), indent=1)
