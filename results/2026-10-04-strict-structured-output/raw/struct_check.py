# Strict structured output checks against the dev engine on 127.0.0.1:8893.
import json, time, urllib.request
def post(body, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:8893/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
def validate(obj, schema):
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict): return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and any(k not in props for k in obj): return False
        if any(k not in obj for k in schema.get("required", [])): return False
        return all(validate(obj[k], props[k]) for k in obj if k in props)
    if t == "array":
        if not isinstance(obj, list): return False
        if "maxItems" in schema and len(obj) > schema["maxItems"]: return False
        if "minItems" in schema and len(obj) < schema["minItems"]: return False
        return all(validate(x, schema.get("items", {})) for x in obj)
    if t == "string": return isinstance(obj, str) and ("enum" not in schema or obj in schema["enum"])
    if t == "integer": return isinstance(obj, int)
    if t == "number": return isinstance(obj, (int, float))
    if t == "boolean": return isinstance(obj, bool)
    return True
def run(name, body, schema):
    t = time.time(); code, d = post(body)
    if code != 200:
        print("%-34s HTTP %s %s" % (name, code, d)); return
    c = d["choices"][0]; m = c["message"]; txt = m.get("content") or ""
    try:
        obj = json.loads(txt); ok = "schema-valid" if (schema is None or validate(obj, schema)) else "SCHEMA-INVALID"
    except Exception:
        ok = "NOT JSON"
    print("%-34s finish=%-6s reasoning=%4d %s chars=%5d tok=%4d %.1fs %r" % (name, c.get("finish_reason"), len(m.get("reasoning_content") or ""), ok, len(txt), d["usage"]["completion_tokens"], time.time() - t, txt[:40]))
small = {"type": "object", "additionalProperties": False, "properties": {"items": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}, "why": {"type": "string"}}, "required": ["name", "why"]}}}, "required": ["items"]}
msg = [{"role": "user", "content": "List three famous bridges and one sentence on why each matters. Answer in JSON."}]
base = {"model": "glm-5.3-flash-atlas", "temperature": 0.25, "max_tokens": 800, "messages": msg}
rf = {"response_format": {"type": "json_schema", "json_schema": {"name": "bridges", "schema": small, "strict": True}}}
for i in range(3):
    run("schema, thinking off #%d" % i, dict(base, chat_template_kwargs={"enable_thinking": False}, **rf), small)
run("schema, effort none", dict(base, reasoning_effort="none", chat_template_kwargs={"enable_thinking": False}, **rf), small)
run("schema, thinking ON", dict(base, chat_template_kwargs={"enable_thinking": True}, **rf), small)
run("json_object, thinking off", dict(base, chat_template_kwargs={"enable_thinking": False}, response_format={"type": "json_object"}), None)
for f in ("rem-capped.json", "rem-uncapped.json"):
    for i in range(2):
        b = json.load(open(f)); l = b.pop("lloom"); sch = l["outputSchema"]; schema = sch.get("schema", sch)
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": sch.get("name") or "rem", "schema": schema, "strict": True}}; b["stream"] = False
        run("%s #%d" % (f, i), b, schema)
# tool call smoke (unchanged path)
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
code, d = post({"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 300, "messages": [{"role": "user", "content": "What's the weather in Paris?"}], "tools": tools, "chat_template_kwargs": {"enable_thinking": False}})
tc = d["choices"][0]["message"].get("tool_calls") if code == 200 else d
print("tool call smoke:", code, json.dumps(tc)[:160])
