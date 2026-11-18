#!/bin/bash
# setup-v7.sh SPARKS [apply]: back up the LLooM config, then install SparkGLM recipe version 7 on this host.
set -u; export PATH=$HOME/.local/bin:$PATH; sparks=${1:?sparks}; mkdir -p ~/sg-036
if [[ ${2:-} == apply ]]; then
  cp -p ~/.lloom/config.json ~/.lloom/config.before-sg036-promo-$(date -u +%Y%m%dT%H%M%SZ).json
  lloom setup --recipe linux-nvidia-dgx-spark-glm53-atlas --recipe-version 7 --sparks $sparks --drafter dflash2 --memory full --streams 4 --additive --apply --yes > ~/sg-036/setup-v7-s$sparks-$(hostname).json 2> ~/sg-036/setup-v7-s$sparks-$(hostname).err; echo "setup rc $?"
else
  lloom setup --recipe linux-nvidia-dgx-spark-glm53-atlas --recipe-version 7 --sparks $sparks --drafter dflash2 --memory full --streams 4 --additive > ~/sg-036/setup-v7-s$sparks-dry-$(hostname).json 2> ~/sg-036/setup-v7-s$sparks-dry-$(hostname).err; echo "dry rc $?"
fi
f=$(ls -t ~/sg-036/setup-v7-s$sparks-*$(hostname).json | head -1); tail -c 300 ${f%.json}.err
python3 - "$f" <<'PY'
import json,sys,os
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("unparsed", e); print(open(sys.argv[1]).read()[-800:]); sys.exit()
print({k:(v if not isinstance(v,(dict,list)) else "…") for k,v in d.items()})
print("recipe", d.get("selectedRecipe"))
for name,ph in (d.get("phases") or {}).items():
    if isinstance(ph,dict): print("phase", name, {k:v for k,v in ph.items() if not isinstance(v,(dict,list))})
c=json.load(open(os.path.expanduser("~/.lloom/config.json")))
for k,r in c["runtimes"].items():
    if k.startswith("glm53-flash-atlas") and "tp4" not in k:
        b=r.get("bootstrap") or {}
        img=[x for x in json.dumps(b).split('"') if "ghcr.io/enntity/atlas-sparkglm" in x][:1]
        print(k, "| recipe", (r.get("recipe") or {}).get("version"), (r.get("recipe") or {}).get("options"), "| mem", r.get("memoryGb"), "| node", r.get("node"), "| maint", (r.get("maintenance") or {}).get("state"), "|", img)
PY
