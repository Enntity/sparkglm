#!/usr/bin/env python3
# mem-conf.py UTIL GB: set the SparkGLM four-Spark runtimes on this host to GPU utilization UTIL and a GB budget.
import json, os, re, sys, tempfile
p = os.path.expanduser("~/.lloom/config.json"); util, gb = sys.argv[1], int(sys.argv[2])
c = json.load(open(p)); n = 0
for k, r in c["runtimes"].items():
    if not k.startswith("glm53-flash-atlas") or "tp4" in k: continue
    if (r.get("recipe") or {}).get("options", {}).get("sparks") != "4": continue
    r["memoryGb"] = gb; n += 1
    for m in (r.get("placement") or {}).get("members", []):
        m.setdefault("resources", {})["memoryGb"] = gb
    b = r.get("bootstrap")
    if b: b["createArgs"] = [re.sub(r"^SPARKGLM_GPU_MEMORY_UTILIZATION=.*", "SPARKGLM_GPU_MEMORY_UTILIZATION=" + util, x) for x in b["createArgs"]]
fd, t = tempfile.mkstemp(dir=os.path.dirname(p)); os.write(fd, json.dumps(c, indent=2).encode()); os.close(fd); os.chmod(t, 0o600); os.replace(t, p)
print(n, "runtimes ->", util, gb)
