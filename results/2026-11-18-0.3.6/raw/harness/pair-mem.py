#!/usr/bin/env python3
# pair-mem.py UTIL GB: set this host's SparkGLM pair runtimes (head, worker, cluster) to GPU utilization UTIL and a GB budget.
# For a host that cannot offer the recipe's 114 GB at admission. Undo: sage-conf.py --restore BACKUP (head and worker), and
# run this again with 0.91 114 for the cluster entry.
import json, os, re, sys, tempfile
p = os.path.expanduser("~/.lloom/config.json"); util, gb = sys.argv[1], int(sys.argv[2])
c = json.load(open(p)); n = 0
for k in ("glm53-flash-atlas-head", "glm53-flash-atlas-worker", "glm53-flash-atlas-cluster"):
    r = c["runtimes"].get(k)
    if not r: continue
    r["memoryGb"] = gb; n += 1
    for m in (r.get("placement") or {}).get("members", []):
        m.setdefault("resources", {})["memoryGb"] = gb
    b = r.get("bootstrap")
    if b: b["createArgs"] = [re.sub(r"^SPARKGLM_GPU_MEMORY_UTILIZATION=.*", "SPARKGLM_GPU_MEMORY_UTILIZATION=" + util, x) for x in b["createArgs"]]
fd, t = tempfile.mkstemp(dir=os.path.dirname(p)); os.write(fd, json.dumps(c, indent=2).encode()); os.close(fd); os.chmod(t, 0o600); os.replace(t, p)
print(n, "runtimes ->", util, gb)
