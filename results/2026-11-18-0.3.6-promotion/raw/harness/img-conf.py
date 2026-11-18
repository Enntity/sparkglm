#!/usr/bin/env python3
# img-conf.py IMAGE: point this host's four-Spark SparkGLM runtimes at IMAGE (already local), leaving everything else.
import json, os, sys, tempfile
p = os.path.expanduser("~/.lloom/config.json"); img = sys.argv[1]; c = json.load(open(p)); n = 0
for k, r in c["runtimes"].items():
    if not k.startswith("glm53-flash-atlas") or "tp4" in k: continue
    if (r.get("recipe") or {}).get("options", {}).get("sparks") != "4": continue
    b = r.get("bootstrap")
    if b: b["image"] = img; b["pull"] = False; n += 1
fd, t = tempfile.mkstemp(dir=os.path.dirname(p)); os.write(fd, json.dumps(c, indent=2).encode()); os.close(fd); os.chmod(t, 0o600); os.replace(t, p)
print(n, "runtimes ->", img[-12:])
