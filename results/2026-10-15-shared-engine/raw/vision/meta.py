#!/usr/bin/env python3
"""meta.py BASE OUT ENGINE IMAGE CONTEXT : RigMark appliance metadata for one arm (BASE with the serving fields replaced)."""
import json, sys
base, out, engine, image, ctx = sys.argv[1:6]
d = json.load(open(base))
d.update({"serving_engine": engine, "serving_image": image, "context_limit": int(ctx), "competing_traffic": "none"})
json.dump(d, open(out, "w"), indent=2)
