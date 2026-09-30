#!/bin/bash
# Merged RC4 core (integ/exp: integ/next + prefill-fifo + startup-parity + dflash-ctx-start + size-caps + decode-repeat
# + warm-ttft), binary ~/spark-rc4, production profile + ATLAS_GLM_WARM_TRACE=hash in every arm.
#   wt-base  no switch: flag-off hash gate, warm-turn probe, min_tokens probe
#   wt-run   + ATLAS_GLM_WARM_SKIP_CACHED=1 ATLAS_GLM_WARM_CHUNK_RUN=1
#   wt-deep  wt-run + ATLAS_GLM_TAIL_CUT_DEEP=1
source $HOME/sparkglm-dev/devlib.sh
IMG=ghcr.io/enntity/atlas-sparkglm:227698449651
cd $D; t0=$(date +%s); el() { echo "[+$(( ($(date +%s) - t0) / 60 )) min] $*"; }
python3 - <<PY
import json
b = json.load(open("pc_on.json")); b["environment"]["ATLAS_GLM_WARM_TRACE"] = "hash"
def w(name, **kv):
    p = json.loads(json.dumps(b)); p["environment"].update(kv); json.dump(p, open(name + ".json", "w"), indent=1)
run = dict(ATLAS_GLM_WARM_SKIP_CACHED="1", ATLAS_GLM_WARM_CHUNK_RUN="1")
w("wt_base"); w("wt_run", **run); w("wt_deep", ATLAS_GLM_TAIL_CUT_DEEP="1", **run)
w("wt_next", ATLAS_PREFILL_SRPT="1", ATLAS_DFLASH_FIRST_APPEND="none", **run)
PY
scp -q wt_*.json $W:$D/
for arm in ${ARMS:-base run deep}; do
  run spark-rc4 wt_$arm || continue; el "wt-$arm up"
  if [ $arm = base ]; then timeout 300 python3 lp_repeat.py rc4-base 2100 1 2>&1 | tail -1; el "base logprob hash"; fi
  timeout 420 python3 warm-ttft-probe.py wt-$arm 2>&1 | tail -3 | cut -c1-300; el "wt-$arm warm probe"
  if [ $arm = base ]; then timeout 300 python3 decode-repeat-probe.py base 2>&1 | tail -4 | cut -c1-300; el "base min_tokens probe"; fi
  logs wt_$arm; cp wt_$arm.rank1.log warm-ttft-wt-$arm.rank1.log
done
python3 - <<PY
import json
try:
    d = json.load(open("$D/lp-rc4-base.json")); h = [r["hash"] for r in d["runs"]]
    print("flag-off logprob hash", h, "== production" if all(x == "8c75d2886794" for x in h) else "DIFFERS FROM PRODUCTION")
except Exception as e: print("logprob missing", e)
PY
prev=wt-base
for arm in ${ARMS:-base run deep}; do [ $arm = base ] && continue; echo "== compare $prev wt-$arm"; python3 warm-ttft-probe.py compare $prev wt-$arm 2>&1 | cut -c1-330 | tail -40; [ $arm = run ] && prev=wt-run; done
echo RC4-DONE
