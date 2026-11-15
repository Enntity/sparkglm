#!/usr/bin/env python3
"""yexact.py OUT.json [--url U] [--dec N] [--dec-out T]: one fixed scenario for the prefill layer yield.
Decoders run greedy while prompts arrive beside them: a cold 36K one, one that forks from it after a shared
prefix, a warm continuation of the first, a 9K one (one full chunk and a tail) and a 5K one (a single pass).
Writes every output, the prompts' token log-probabilities, each prompt's TTFT and the decoders' gaps while
each prompt prefilled. Run it on a fresh engine with the yield off and again with it on, then
`yexact.py --cmp A.json B.json`: outputs and log-probabilities must be identical."""
import argparse, json, sys, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("b", nargs="?")
ap.add_argument("--cmp", action="store_true"); ap.add_argument("--url", default="http://127.0.0.1:8893")
ap.add_argument("--dec", type=int, default=2); ap.add_argument("--dec-out", type=int, default=6000)
ap.add_argument("--model", default="glm-5.3-flash-atlas"); ap.add_argument("--json-dec", type=int, default=1)
ap.add_argument("--lp", action="store_true", help="prompts unstreamed with token log-probabilities (no TTFT)")
ap.add_argument("--salt", type=int, default=0, help="a different set of prompts, cold on an engine that ran another salt")
a = ap.parse_args()
if a.cmp:
    A, B = json.load(open(a.out)), json.load(open(a.b)); bad = 0
    for k in sorted(A["runs"]):
        x, y = A["runs"][k], B["runs"].get(k)
        if y is None: print(k, "MISSING in B"); bad += 1; continue
        same_text = x["text"] == y["text"]; same_lp = x.get("logprobs") == y.get("logprobs")
        if not same_text:
            i = next((i for i, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q), min(len(x["text"]), len(y["text"])))
            print(f"{k}: TEXT DIFFERS at char {i}: {x['text'][max(0,i-40):i+40]!r} vs {y['text'][max(0,i-40):i+40]!r}"); bad += 1
        elif not same_lp:
            lx, ly = x.get("logprobs") or [], y.get("logprobs") or []
            i = next((i for i, (p, q) in enumerate(zip(lx, ly)) if p != q), min(len(lx), len(ly)))
            print(f"{k}: LOGPROBS DIFFER at token {i}: {lx[i:i+3]} vs {ly[i:i+3]}"); bad += 1
        else: print(f"{k}: identical ({x['tokens']} tokens, {len(x.get('logprobs') or [])} logprobs)")
        print(f"   ttft {x['ttft']:.2f} -> {y['ttft']:.2f} s   beside decoders: {x.get('beside')} / {y.get('beside')}"
              + (f"   decoder max gap {x['dec_gap']:.2f} -> {y['dec_gap']:.2f} s, decoder tokens meanwhile {x['dec_tokens']} -> {y['dec_tokens']}" if "dec_gap" in x else ""))
    print("RESULT:", "IDENTICAL" if not bad else f"{bad} DIFFER"); sys.exit(1 if bad else 0)

WORDS = ("harbor lantern granite willow ledger compass orchard signal copper meadow archive tunnel velvet ember quartz "
         "bridge candle furnace garden hollow island jacket kettle lumber magnet needle outpost pebble quarry ribbon").split()
def notes(n_tokens, seed):
    seed += a.salt * 100003
    out, i, x = [], 0, seed * 7919 + 13
    while len(out) * 1.35 < n_tokens:
        x = (x * 1103515245 + 12345) % 2**31; out.append(WORDS[x % len(WORDS)]); i += 1
        if i % 17 == 0: out.append(f"\nEntry {i // 17} ({seed}):")
    return " ".join(out)
SYS = {"role": "system", "content": "You are a careful archivist."}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "minItems": 40, "maxItems": 40,
          "items": {"type": "object", "additionalProperties": False, "required": ["entry", "summary", "mood"],
                    "properties": {"entry": {"type": "integer"}, "summary": {"type": "string"}, "mood": {"type": "string", "enum": ["calm", "tense", "bright", "flat"]}}}}}}
def req(messages, out, logprobs=False, schema=False):
    b = {"model": a.model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": out, "messages": messages}
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "appraisals", "strict": True, "schema": SCHEMA}}
        b.update(reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
    else: b["reasoning_effort"] = "low"
    if logprobs and a.lp: b.update(logprobs=True, top_logprobs=1, stream=False); b.pop("stream_options")
    return b
def run(b, rec):
    r = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json", "authorization": "Bearer sk-local-llm"})
    rec.update(t0=time.time(), stamps=[], text="", reason="", logprobs=[], tokens=0, done=False)
    try:
        with urllib.request.urlopen(r, timeout=3600) as h:
            if not b["stream"]:
                j = json.load(h); c = j["choices"][0]; m = c["message"]
                rec.update(tokens=j["usage"].get("completion_tokens", 0), prompt=j["usage"].get("prompt_tokens"), reason=m.get("reasoning_content") or "", text=m.get("content") or "",
                           logprobs=[[e.get("token"), e.get("logprob")] for e in ((c.get("logprobs") or {}).get("content") or [])], stamps=[time.time()])
            for line in h:
                if not line.startswith(b"data:"): continue
                d = line[5:].strip()
                if d == b"[DONE]": break
                j = json.loads(d)
                if j.get("usage"): rec["tokens"] = j["usage"].get("completion_tokens", 0); rec["prompt"] = j["usage"].get("prompt_tokens")
                c = (j.get("choices") or [{}])[0]; ch = c.get("delta") or {}
                for e in ((c.get("logprobs") or {}).get("content") or []): rec["logprobs"].append([e.get("token"), e.get("logprob")])
                piece = (ch.get("reasoning_content") or ch.get("reasoning") or "", ch.get("content") or "")
                if piece[0] or piece[1]:
                    rec["stamps"].append(time.time()); rec["reason"] += piece[0]; rec["text"] += piece[1]
    except Exception as e: rec["error"] = repr(e)
    rec["t1"] = time.time(); rec["done"] = True
decs = []
for i in range(a.dec):
    js = i < a.json_dec
    task = ("Pick forty entries from the notes and record an appraisal of each: a one-sentence summary of at least twenty words and a mood." if js
            else "Write a very long reflection on the notes above, one detailed paragraph per entry, quoting entry numbers. Do not stop early.")
    rec = {}; b = req([SYS, {"role": "user", "content": f"Notes (dec-{i}):\n{notes(700, 500 + i)}\n\n{task}"}], a.dec_out, schema=js)
    t = threading.Thread(target=run, args=(b, rec)); t.start(); decs.append((t, rec)); time.sleep(1.5)
time.sleep(4)
shared = notes(4200, 7001)
P1 = [SYS, {"role": "user", "content": f"Notes (p1):\n{shared}\n{notes(31000, 7002)}\n\nName the three most frequent words in the notes and say why in two sentences."}]
runs = {}
def prompt(name, messages, out=96):
    rec = {}; run(req(messages, out, logprobs=True), rec)
    first = rec["stamps"][0] if rec["stamps"] else rec["t1"]
    w = [(s, d) for _, d in decs for s in d["stamps"] if rec["t0"] <= s <= first]
    gaps = []
    for _, d in decs:
        st = [rec["t0"]] + [s for s in d["stamps"] if rec["t0"] <= s <= first] + [first]
        gaps.append(max(q - p for p, q in zip(st, st[1:])))
    runs[name] = dict(text=rec["reason"] + "\x00" + rec["text"], logprobs=rec["logprobs"], tokens=rec["tokens"], prompt=rec.get("prompt"), ttft=first - rec["t0"],
                      error=rec.get("error"), beside=sum(1 for _, d in decs if not d["done"]), dec_gap=max(gaps), dec_tokens=len(w))
    print(name, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in runs[name].items() if k not in ("text", "logprobs")}), flush=True)
    return rec
r1 = prompt("p1-cold-36k", P1)
prompt("p2-fork-20k", [SYS, {"role": "user", "content": f"Notes (p1):\n{shared}\n{notes(15000, 7003)}\n\nName the three least frequent words in the notes."}])
prompt("p3-warm-turn", P1 + [{"role": "assistant", "content": r1["text"]},
                              {"role": "user", "content": f"More notes:\n{notes(3200, 7004)}\n\nDo the new notes change your answer? Two sentences."}])
prompt("p4-cold-9k", [SYS, {"role": "user", "content": f"Notes (p4):\n{notes(8600, 7005)}\n\nWhich entry number appears last? One sentence."}])
prompt("p5-cold-5k", [SYS, {"role": "user", "content": f"Notes (p5):\n{notes(4600, 7006)}\n\nWhich word opens the notes? One sentence."}])
for i, (t, d) in enumerate(decs):
    t.join()
    runs[f"dec-{i}"] = dict(text=d["reason"] + "\x00" + d["text"], tokens=d["tokens"], ttft=d["stamps"][0] - d["t0"] if d["stamps"] else -1, error=d.get("error"),
                           wall=d["t1"] - d["t0"])
    print(f"dec-{i}", d["tokens"], "tokens", round(d["t1"] - d["t0"], 1), "s", d.get("error") or "")
json.dump({"runs": runs}, open(a.out, "w"))
