import json, urllib.request
schema = {"type": "object", "additionalProperties": False, "properties": {"items": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"name": {"type": "string"}, "why": {"type": "string"}}, "required": ["name", "why"]}}}, "required": ["items"]}
b = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 800, "chat_template_kwargs": {"enable_thinking": False},
     "messages": [{"role": "user", "content": "List three famous bridges and one sentence on why each matters. Answer in JSON."}],
     "response_format": {"type": "json_schema", "json_schema": {"name": "bridges", "schema": schema, "strict": True}}}
d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8893/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=300))
m = d["choices"][0]["message"]
print("REASONING:", repr((m.get("reasoning_content") or "")[:400]))
print("CONTENT:", repr((m.get("content") or "")[:200]))
print("finish:", d["choices"][0]["finish_reason"], "usage:", d["usage"])
