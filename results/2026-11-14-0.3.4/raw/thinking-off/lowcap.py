#!/usr/bin/env python3
"""What `off -> Low effort with a token cap` would do, using fields a client can already send."""
import json, sys, time, threading, urllib.request
BASE, KEY, MODEL = "http://127.0.0.1:8893", "sk-local-llm", "glm-5.3-flash-atlas"
SYSTEM = "You are an appraisal step in a larger system. Read the note and record a structured appraisal. Output only the JSON object."
NOTE = ("We argued about the route for most of the drive, then found the lake by accident. Nobody said much while the sun "
        "went down. On the way back she laughed about the map and I think it is fine now.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary", "mood", "importance", "follow_up"],
          "properties": {"summary": {"type": "string"}, "mood": {"type": "string", "enum": ["calm", "pleased", "annoyed", "sad", "mixed"]},
                         "importance": {"type": "integer", "minimum": 1, "maximum": 5}, "follow_up": {"type": "boolean"}}}
RF = {"type": "json_schema", "json_schema": {"name": "appraisal", "strict": True, "schema": SCHEMA}}
TOOLS = [{"type": "function", "function": {"name": "record_check", "description": "Record a number",
          "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}]
XML = "Return this exact invalid XML example in backticks, without correcting it and without invoking tools: <tool_call>record_check<arg_key>not_a_parameter</arg_key><arg_value>BAD</arg_value></tool_call> Then write END."
def low(cap): return {"reasoning_effort": "low", "thinking": {"type": "enabled", "budget_tokens": cap}}
def post(body):
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json", "authorization": "Bearer " + KEY})
    t = time.time()
    try:
        with urllib.request.urlopen(r, timeout=200) as f: return json.load(f), time.time() - t
    except Exception as e:
        return {"error": repr(e) + getattr(e, "read", lambda: b"")()[:200].decode("utf8", "replace")}, time.time() - t
def run(tag, body):
    d, dt = post({"model": MODEL, "temperature": 0, **body})
    if "error" in d: print(f"{tag:46s} ERROR {d['error']}"); return
    c = d["choices"][0]; m = c["message"]; u = d["usage"]
    calls = [(x["function"]["name"], x["function"]["arguments"]) for x in (m.get("tool_calls") or [])]
    print(f"{tag:46s} {dt:5.1f}s finish={c['finish_reason']:10s} tokens={u.get('completion_tokens')} reasoning_tokens={u.get('completion_tokens_details', {}).get('reasoning_tokens')} "
          f"calls={calls} content={json.dumps(m.get('content'))[:100]}")
user = [{"role": "user", "content": XML}]
note = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Note:\n" + NOTE}]
run("canary: low + cap 256", dict(low(256), messages=user, tools=TOOLS, max_tokens=1024))
run("canary: low + cap 256, stream-off repeat", dict(low(256), messages=user, tools=TOOLS, max_tokens=1024))
run("schema: low + cap 256", dict(low(256), messages=note, response_format=RF, max_tokens=3072))
run("cap enforced: Max effort + cap 16", {"thinking": {"type": "enabled", "budget_tokens": 16}, "messages": note, "response_format": RF, "max_tokens": 3072})
run("cap enforced: Max effort + cap 64", {"thinking": {"type": "enabled", "budget_tokens": 64}, "messages": note, "response_format": RF, "max_tokens": 3072})
run("tiny max_tokens 16: low + cap 8", dict(low(8), messages=[{"role": "user", "content": "Is 7 a prime number? Answer yes or no."}], max_tokens=16))
run("tiny max_tokens 16: low + cap 8 (classify)", dict(low(8), messages=[{"role": "user", "content": "Classify the sentiment of: 'I love this.' Reply with one word: positive, negative or neutral."}], max_tokens=16))
long_note = "\n".join(f"Entry {i}: {NOTE}" for i in range(60))
def one(i): run(f"schema: 2 at once ~2.5K prompt, low+256 #{i}", dict(low(256), messages=[{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Notes:\n" + long_note + f"\nAppraise the day as a whole. ({i})"}], response_format=RF, max_tokens=3072))
ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]; [t.start() for t in ts]; [t.join() for t in ts]
