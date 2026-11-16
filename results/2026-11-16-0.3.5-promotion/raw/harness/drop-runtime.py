#!/usr/bin/env python3
# drop-runtime.py NAME...: remove runtime entries that a recipe re-install left behind, atomically.
import json, os, sys, tempfile
p = os.path.expanduser("~/.lloom/config.json"); c = json.load(open(p))
for name in sys.argv[1:]:
    r = c["runtimes"].pop(name, None)
    if r is None: print("absent:", name); continue
    refs = [k for k, v in c["runtimes"].items() if '"%s"' % name in json.dumps(v)]
    if refs: sys.exit("refused: %s is still referenced by %s" % (name, refs))
    print("removed:", name, "node", r.get("node"), "recipe", (r.get("recipe") or {}).get("version"), (r.get("recipe") or {}).get("options"))
fd, t = tempfile.mkstemp(dir=os.path.dirname(p)); os.write(fd, json.dumps(c, indent=2).encode()); os.close(fd); os.chmod(t, 0o600); os.replace(t, p)
