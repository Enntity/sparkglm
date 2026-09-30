#!/usr/bin/env python3
"""lp_repeat.py TAG [ROWS=2100] [REPEAT=3]: prompt-logprob hash of one long prompt, REPEAT times in one server start.
Scoring requests recompute the whole prompt (no prefix-cache reuse), so equal hashes mean prefill is reproducible."""
import hashlib, json, random, sys, time, urllib.request
URL = "http://127.0.0.1:8893"; MODEL = "glm-5.3-flash-atlas"
def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2100
rng = random.Random(9090)
text = "\n".join(f"Record {i}: unit-{i:05d} has access code {rng.randint(100000,999999)} and owner team-{rng.randint(1,99)}." for i in range(rows))
out = []
for r in range(int(sys.argv[3]) if len(sys.argv) > 3 else 3):
    t = time.time()
    c = post("/v1/completions", {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1})
    vals = [x for x in c["choices"][0]["logprobs"]["token_logprobs"] if x is not None]
    out.append({"n": len(vals), "hash": hashlib.sha1(json.dumps(vals).encode()).hexdigest()[:12], "sum": round(sum(vals), 4), "secs": round(time.time() - t, 1)})
    print(out[-1], flush=True)
same = len({o["hash"] for o in out}) == 1
json.dump({"tag": sys.argv[1], "runs": out, "reproducible": same}, open(f"lp-{sys.argv[1]}.json", "w"), indent=1)
print("REPRODUCIBLE" if same else "DIFFERS", sys.argv[1])
