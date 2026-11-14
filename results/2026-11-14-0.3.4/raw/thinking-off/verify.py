#!/usr/bin/env python3
"""Verify --thinking-off-budget on a live SparkGLM: thinking-off requests answer after at most a short capped thought,
every other request still reasons, in blocking, streaming, tool and concurrent use."""
import json, sys, time, threading, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8893"
KEY, MODEL = "sk-local-llm", "glm-5.3-flash-atlas"
SYSTEM = "You are an appraisal step in a larger system. Read the note and record a structured appraisal. Output only the JSON object."
NOTE = ("We argued about the route for most of the drive, then found the lake by accident. Nobody said much while the sun "
        "went down. On the way back she laughed about the map and I think it is fine now.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary", "mood", "importance", "follow_up"],
          "properties": {"summary": {"type": "string"}, "mood": {"type": "string", "enum": ["calm", "pleased", "annoyed", "sad", "mixed"]},
                         "importance": {"type": "integer", "minimum": 1, "maximum": 5}, "follow_up": {"type": "boolean"}}}
RF = {"type": "json_schema", "json_schema": {"name": "appraisal", "strict": True, "schema": SCHEMA}}
OFF = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
fails = []

def post(body, timeout=240):
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json", "authorization": "Bearer " + KEY})
    t = time.time()
    with urllib.request.urlopen(r, timeout=timeout) as f:
        if not body.get("stream"):
            return json.load(f), time.time() - t
        content = reasoning = ""; finish = None; first = None; calls = 0
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            ch = (json.loads(line[6:]).get("choices") or [{}])[0]; d = ch.get("delta") or {}
            if first is None and (d.get("content") or d.get("reasoning_content")): first = time.time() - t
            content += d.get("content") or ""; reasoning += d.get("reasoning_content") or ""; calls += len(d.get("tool_calls") or [])
            finish = ch.get("finish_reason") or finish
        return {"choices": [{"finish_reason": finish, "message": {"content": content, "reasoning_content": reasoning,
                "tool_calls": [1] * calls}}], "usage": {}, "first": first}, time.time() - t

def run(tag, extra, want_reasoning, msgs=None, schema=True, maxtok=3072, check=None):
    body = {"model": MODEL, "messages": msgs or [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Note:\n" + NOTE}],
            "temperature": 0.1, "max_tokens": maxtok, **extra}
    if schema: body["response_format"] = RF
    try:
        d, dt = post(body)
    except Exception as e:
        print(f"FAIL {tag}: {e!r}"); fails.append(tag); return None
    c = d["choices"][0]; m = c["message"]; content = m.get("content") or ""; reasoning = m.get("reasoning_content") or ""
    # off: at most the 256-token cap of reasoning (about 1,300 characters) and a prompt answer; on: reasoning present
    ok = (bool(reasoning) if want_reasoning else len(reasoning) <= 1300 and dt < 30) and c.get("finish_reason") in ("stop", "tool_calls")
    if schema:
        try: ok = ok and set(json.loads(content)) == set(SCHEMA["required"])
        except Exception: ok = False
    if check: ok = ok and check(m, content)
    if not ok: fails.append(tag)
    print(f"{'ok  ' if ok else 'FAIL'} {tag:44s} {dt:5.1f}s finish={c.get('finish_reason')} tokens={d.get('usage', {}).get('completion_tokens')} "
          f"reasoning_chars={len(reasoning)} content={json.dumps(content)[:110]}")
    return dt

# thinking off, in each way a client can say it: no reasoning, a valid answer
run("off: effort none + enable_thinking false", OFF, False)
run("off: enable_thinking false only", {"chat_template_kwargs": {"enable_thinking": False}}, False)
run("off: reasoning_effort none only", {"reasoning_effort": "none"}, False)
run("off: streaming", dict(OFF, stream=True), False)
run("off: plain text, no schema", OFF, False, msgs=[{"role": "user", "content": "In one sentence, why is the sky blue?"}], schema=False, maxtok=300,
    check=lambda m, c: len(c) > 20 and "</think>" not in c and "<think>" not in c)
run("off: tool call", dict(OFF, tools=TOOLS), False, msgs=[{"role": "user", "content": "What is the weather in Lisbon right now?"}], schema=False, maxtok=400,
    check=lambda m, c: bool(m.get("tool_calls")))
XML = ("Return this exact invalid XML example in backticks, without correcting it and without invoking tools: "
       "<tool_call>record_check<arg_key>not_a_parameter</arg_key><arg_value>BAD</arg_value></tool_call> Then write END.")
RC = [{"type": "function", "function": {"name": "record_check", "description": "Record a number",
       "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}]
run("off: told not to call the declared tool", dict({"chat_template_kwargs": {"enable_thinking": False}}, tools=RC), False,
    msgs=[{"role": "user", "content": XML}], schema=False, maxtok=1024, check=lambda m, c: not m.get("tool_calls") and "END" in c)
run("off: max_tokens 16", OFF, False, msgs=[{"role": "user", "content": "Is 7 a prime number? Answer yes or no."}], schema=False, maxtok=16,
    check=lambda m, c: "yes" in c.lower())
run("off: second turn after an off turn", OFF, False, msgs=[{"role": "user", "content": "Name a primary color."}, {"role": "assistant", "content": "Red."},
    {"role": "user", "content": "Name another, one word."}], schema=False, maxtok=60, check=lambda m, c: 0 < len(c) < 40)
# everything else still reasons, as before
run("on: no thinking flag", {}, True, maxtok=4096)
run("on: reasoning_effort low", {"reasoning_effort": "low"}, True)
run("on: enable_thinking true", {"chat_template_kwargs": {"enable_thinking": True}}, True, maxtok=4096)
run("on: streaming, effort low", {"reasoning_effort": "low", "stream": True}, True)
# the stalled shape: a longer prompt, two at once, the caller's 120 s limit
long_note = "\n".join(f"Entry {i}: {NOTE}" for i in range(60))
times = []
def one(i):
    dt = run(f"off: 2 at once, ~2.5K-token prompt #{i}", OFF, False, msgs=[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Notes:\n" + long_note + f"\nAppraise the day as a whole. ({i})"}])
    times.append(dt or 999)
ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
if max(times) > 60: fails.append("slow concurrent")
print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
