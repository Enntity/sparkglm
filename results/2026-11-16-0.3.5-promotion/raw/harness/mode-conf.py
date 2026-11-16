#!/usr/bin/env python3
# mode-conf.py IMAGE MODE: point this host SparkGLM pair runtimes at IMAGE and SPARKGLM_MODE=MODE.
import json, os, re, sys
p = os.path.expanduser("~/.lloom/config.json"); img, mode = sys.argv[1], sys.argv[2]
c = json.load(open(p)); n = 0
for k in ("glm53-flash-atlas-head", "glm53-flash-atlas-worker"):
    b = c["runtimes"].get(k, {}).get("bootstrap")
    if not b: continue
    b["image"] = img; b["pull"] = False; n += 1
    b["createArgs"] = [re.sub(r"^SPARKGLM_MODE=.*", "SPARKGLM_MODE=" + mode, x) for x in b["createArgs"]]
json.dump(c, open(p, "w"), indent=2); print(n, "runtimes ->", img, mode)
