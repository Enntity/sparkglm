#!/usr/bin/env python3
"""The literal-tool-XML canary (tools declared, told not to call them) under each thinking setting."""
import json, sys, time, urllib.request
BASE, KEY, MODEL = "http://127.0.0.1:8893", "sk-local-llm", "glm-5.3-flash-atlas"
TOOLS = [{"type": "function", "function": {"name": "record_check", "description": "Record a number",
          "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}]
PROMPTS = {
 "literal-xml": "Return this exact invalid XML example in backticks, without correcting it and without invoking tools: <tool_call>record_check<arg_key>not_a_parameter</arg_key><arg_value>BAD</arg_value></tool_call> Then write END.",
 "no-tool-needed": "Without using any tool, tell me in one sentence what record_check is for.",
 "tool-needed": "Record the number 41 using the tool.",
 "plain-question": "What is 17 + 26? Answer with the number only.",
}
ARMS = {"off": {"chat_template_kwargs": {"enable_thinking": False}}, "low": {"reasoning_effort": "low"}, "max": {}}
def post(body):
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json", "authorization": "Bearer " + KEY})
    t = time.time()
    with urllib.request.urlopen(r, timeout=200) as f: return json.load(f), time.time() - t
for pname, prompt in PROMPTS.items():
    for arm in sys.argv[1].split(","):
        d, dt = post({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024, "temperature": 0, "tools": TOOLS, **ARMS[arm]})
        c = d["choices"][0]; m = c["message"]
        calls = [(x["function"]["name"], x["function"]["arguments"]) for x in (m.get("tool_calls") or [])]
        print(f"{pname:15s} {arm:4s} {dt:5.1f}s finish={c['finish_reason']:10s} reasoning_tokens={d['usage'].get('completion_tokens_details', {}).get('reasoning_tokens')} calls={calls} content={json.dumps(m.get('content'))[:120]}")
