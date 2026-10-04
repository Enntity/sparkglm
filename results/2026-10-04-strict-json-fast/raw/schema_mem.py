# Peak host memory drop while a strict object schema with N keys compiles and decodes (MemAvailable sampled every 0.2 s).
import json, sys, time, threading, urllib.request
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) // 1024
def run(n):
    keys = ["m%03d" % i for i in range(n)]
    schema = {"type": "object", "additionalProperties": False, "required": keys,
              "properties": {k: {"type": "object", "additionalProperties": False, "required": ["keep", "reason"],
                                 "properties": {"keep": {"type": "boolean"}, "reason": {"type": "string"}}} for k in keys}}
    prompt = "For each id decide keep true/false with a short reason.\n" + "\n".join("%s: topic %d" % (k, i) for i, k in enumerate(keys))
    b = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 40 * n, "chat_template_kwargs": {"enable_thinking": False},
         "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema, "strict": True}}}
    base = avail(); low = [base]; stop = [False]
    def sampler():
        while not stop[0]:
            low[0] = min(low[0], avail()); time.sleep(0.2)
    th = threading.Thread(target=sampler); th.start(); t = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8893/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
        res = "finish=%s tokens=%d" % (d["choices"][0]["finish_reason"], d["usage"]["completion_tokens"])
    except Exception as e:
        res = "FAILED %s" % str(e)[:60]
    stop[0] = True; th.join()
    print("keys=%3d start %6d MiB  min %6d MiB  drop %6d MiB  %.0fs  %s" % (n, base, low[0], base - low[0], time.time() - t, res), flush=True)
    return low[0]
for n in [int(x) for x in sys.argv[1:]]:
    if run(n) < 3000:
        print("stopping: headroom below 3 GiB"); break
