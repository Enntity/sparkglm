# 96-key strict object at temperature 0, and strict + unconstrained concurrency.
import json, time, urllib.request, threading
URL = "http://127.0.0.1:8893/v1/chat/completions"
def post(b, timeout=1800):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=timeout))
keys = ["m%03d" % i for i in range(96)]
schema = {"type": "object", "additionalProperties": False, "required": keys,
          "properties": {k: {"type": "object", "additionalProperties": False, "required": ["keep", "reason"],
                             "properties": {"keep": {"type": "boolean"}, "reason": {"type": "string"}}} for k in keys}}
prompt = "For each memory id below decide keep true/false with a short reason (under 12 words).\n" + "\n".join("%s: note about topic %d, mentioned %d times" % (k, i, i % 7) for i, k in enumerate(keys))
b = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 6000, "chat_template_kwargs": {"enable_thinking": False},
     "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_schema", "json_schema": {"name": "recon", "schema": schema, "strict": True}}}
t = time.time(); d = post(b); c = d["choices"][0]; txt = c["message"]["content"] or ""
try:
    o = json.loads(txt); ok = "valid, %d keys, all present: %s" % (len(o), set(o) == set(keys))
except Exception as e:
    ok = "INVALID %s" % str(e)[:50]
n = d["usage"]["completion_tokens"]; w = time.time() - t
print("96-key strict t=0: finish=%s tokens=%d %.1fs %.1f tok/s %s" % (c["finish_reason"], n, w, n / w, ok))
# concurrency: one strict REM-sized request + one plain prose request at the same time
small = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {"type": "string"}}}}
strict = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 1200, "chat_template_kwargs": {"enable_thinking": False},
          "messages": [{"role": "user", "content": "List 60 short facts about the ocean as JSON."}], "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "schema": small, "strict": True}}}
plain = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 600, "min_tokens": 600, "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": "Explain how a hash map works, in flowing prose. No code, no lists."}]}
res = {}
def go(name, body):
    t = time.time(); d = post(body); res[name] = (d["usage"]["completion_tokens"], time.time() - t, d["choices"][0]["finish_reason"])
th = [threading.Thread(target=go, args=a) for a in (("strict", strict), ("plain", plain))]
[x.start() for x in th]; [x.join() for x in th]
for k, (n, w, f) in res.items(): print("concurrent %-6s tokens=%d %.1fs %.1f tok/s finish=%s" % (k, n, w, n / w, f))
t = time.time(); d = post(plain); n = d["usage"]["completion_tokens"]; w = time.time() - t
print("plain alone          tokens=%d %.1fs %.1f tok/s" % (n, w, n / w))
