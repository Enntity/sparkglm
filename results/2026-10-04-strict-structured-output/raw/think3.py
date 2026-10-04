import json, urllib.request
def chat(extra):
    b = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 120, "messages": [{"role": "user", "content": "Name one famous bridge in a short sentence."}]}
    b.update(extra)
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8893/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=120))
    m = d["choices"][0]["message"]
    return len(m.get("reasoning_content") or ""), (m.get("content") or "")[:60], d["usage"]["completion_tokens_details"].get("reasoning_tokens")
schema = {"type": "object", "additionalProperties": False, "properties": {"bridge": {"type": "string"}}, "required": ["bridge"]}
rf = {"response_format": {"type": "json_schema", "json_schema": {"name": "b", "schema": schema, "strict": True}}}
for name, extra in [("thinking off, no schema", {"chat_template_kwargs": {"enable_thinking": False}}),
                    ("thinking off + schema", dict(chat_template_kwargs={"enable_thinking": False}, **rf)),
                    ("thinking on, no schema", {"chat_template_kwargs": {"enable_thinking": True}}),
                    ("json_object, thinking off", {"chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"}})]:
    print("%-28s reasoning_chars=%-4s reasoning_tokens=%-4s content=%r" % ((name,) + chat(extra)[:1] + (chat(extra)[2],) + (chat(extra)[1],)))
