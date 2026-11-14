#!/usr/bin/env python3
"""soak.py [--secs 150] [--streams 4]: mixed concurrent traffic (strict JSON, free text, tool calls, sampled JSON, streaming JSON)
for a while; every JSON answer must parse and satisfy its schema's required keys, every tool request must return the tool call."""
import argparse, json, random, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--secs", type=int, default=150); ap.add_argument("--streams", type=int, default=4)
ap.add_argument("--url", default="http://127.0.0.1:8893"); ap.add_argument("--only", default=""); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
SCH = {"type": "object", "additionalProperties": False, "required": ["city", "country", "facts"], "properties": {"city": {"type": "string"}, "country": {"type": "string"},
       "facts": {"type": "array", "minItems": 3, "maxItems": 5, "items": {"type": "object", "additionalProperties": False, "required": ["topic", "detail", "year"],
                 "properties": {"topic": {"type": "string", "enum": ["history", "food", "geography", "culture"]}, "detail": {"type": "string"}, "year": {"type": "integer"}}}}}}
RF = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": SCH}}
TOOLS = [{"type": "function", "function": {"name": "record_city", "description": "Record a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}}, "required": ["city"]}}}]
CITIES = "Lisbon Osaka Nairobi Lima Oslo Hanoi Quito Accra Perth Turin Leeds Cork Riga Bergen Porto Kyoto".split()
stats = {"json": [0, 0], "json_stream": [0, 0], "json_sampled": [0, 0], "text": [0, 0], "tool": [0, 0], "json_object": [0, 0]}; errs = []; lock = threading.Lock()
def post(b):
    r = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(b).encode(), {"content-type": "application/json", "authorization": "Bearer sk-local-llm"})
    with urllib.request.urlopen(r, timeout=300) as f:
        if not b.get("stream"): return json.load(f)["choices"][0]
        content = ""; fin = None; rc = ""
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            ch = (json.loads(line[6:]).get("choices") or [{}])[0]; content += (ch.get("delta") or {}).get("content") or ""; rc += (ch.get("delta") or {}).get("reasoning_content") or ""; fin = ch.get("finish_reason") or fin
        return {"finish_reason": fin, "message": {"content": content, "reasoning_content": rc}}
def valid(s):
    d = json.loads(s); assert set(d) == set(SCH["required"]) and 3 <= len(d["facts"]) <= 5
    for f in d["facts"]: assert set(f) == {"topic", "detail", "year"} and f["topic"] in ("history", "food", "geography", "culture") and isinstance(f["year"], int)
def worker(k):
    rnd = random.Random(k + a.seed * 1000); end = time.time() + a.secs
    while time.time() < end:
        kind = rnd.choice([x for x in stats if not a.only or x in a.only.split(',')]); city = rnd.choice(CITIES); n = rnd.randint(1, 10**6)
        b = {"model": "glm-5.3-flash-atlas", "max_tokens": 700, "temperature": 0, "messages": [{"role": "user", "content": f"({n}) Tell me about {city}."}]}
        off = dict(reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
        if kind == "json": b.update(response_format=RF, **off)
        elif kind == "json_stream": b.update(response_format=RF, stream=True, reasoning_effort="low")
        elif kind == "json_sampled": b.update(response_format=RF, temperature=0.8, seed=n, **off)
        elif kind == "json_object": b.update(response_format={"type": "json_object"}, **off); b["messages"][0]["content"] += " Answer as a JSON object with keys city and country."
        elif kind == "tool": b.update(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "record_city"}}, reasoning_effort="low")
        else: b.update(reasoning_effort="low", max_tokens=250)
        ok = False; c = {}
        try:
            c = post(b); m = c["message"]
            if kind in ("json", "json_stream", "json_sampled"): valid(m["content"]); ok = c["finish_reason"] == "stop"
            elif kind == "json_object": json.loads(m["content"]); ok = c["finish_reason"] == "stop"
            elif kind == "tool": ok = bool(m.get("tool_calls")) and "city" in json.loads(m["tool_calls"][0]["function"]["arguments"])
            else: ok = len(m.get("content") or "") > 20
            if not ok: raise AssertionError(f"finish={c['finish_reason']} content={str(m.get('content'))[:120]!r}")
        except Exception as e:
            mm = c.get("message") or {}
            with lock: errs.append(f"{kind}: {e!r} finish={c.get('finish_reason')} content_chars={len(mm.get('content') or '')} reasoning_chars={len(mm.get('reasoning_content') or '')} prompt={b['messages'][0]['content'][:40]!r}"[:330])
        with lock: stats[kind][0] += 1; stats[kind][1] += ok
th = [threading.Thread(target=worker, args=(k,)) for k in range(a.streams)]; [t.start() for t in th]; [t.join() for t in th]
print({k: f"{v[1]}/{v[0]}" for k, v in stats.items()}); [print("  ERR", e) for e in errs[:8]]
print("RESULT:", "PASS" if not errs else f"FAIL ({len(errs)})")
