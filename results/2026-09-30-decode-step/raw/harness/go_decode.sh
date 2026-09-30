#!/bin/bash
# Decode-step campaign binary (integ/decode = integ/next + six branches), ~/spark-decode. Every arm has ATLAS_DFLASH_FIRST_APPEND=none
# so greedy text is history-independent and comparable across arms.
#   base   no new flag (flag-off hash gate)
#   exact  CMD_RDMA + MOE_DECODE_STREAM + DRAFT_TP + DECODE_FUSE + DECODE_GEMV_BATCH (all exact: same text, same tok/step)
#   conf   exact + DFLASH_CONF_WIDTH (output exact per draft; changes rows verified)
source $HOME/sparkglm-dev/devlib.sh
IMG=ghcr.io/enntity/atlas-sparkglm:227698449651
cd $D; t0=$(date +%s); el() { echo "[+$(( ($(date +%s) - t0) / 60 )) min] $*"; }
python3 - <<PY
import json
b = json.load(open("pc_on.json")); b["environment"]["ATLAS_DFLASH_FIRST_APPEND"] = "none"
def w(name, **kv):
    p = json.loads(json.dumps(b)); p["environment"].update(kv); json.dump(p, open(name + ".json", "w"), indent=1)
ex = dict(ATLAS_GLM_CMD_RDMA="1", ATLAS_GLM_MOE_DECODE_STREAM="1", ATLAS_GLM_DRAFT_TP="1", ATLAS_GLM_DECODE_FUSE="1", ATLAS_GLM_DECODE_GEMV_BATCH="1")
w("dc_base"); w("dc_exact", **ex); w("dc_conf", ATLAS_DFLASH_CONF_WIDTH="1", **ex)
for k, v in ex.items(): w("dc_only_" + k.split("ATLAS_GLM_")[1].lower(), **{k: v})
PY
scp -q dc_*.json $W:$D/
for arm in ${ARMS:-base exact conf}; do
  run spark-decode dc_$arm || continue; el "$arm up"
  if [ $arm = base ]; then timeout 300 python3 lp_repeat.py dc-base 2100 1 2>&1 | tail -1; fi
  timeout 330 python3 vg-probe-nx.py dc$arm 2>&1 | tail -1 | cut -c1-300; el "$arm decode probe"
  logs dc_$arm
done
python3 - <<PY
import json, os
try:
    d = json.load(open("$D/lp-dc-base.json")); h = [r["hash"] for r in d["runs"]]
    print("flag-off logprob hash", h, "== production" if all(x == "8c75d2886794" for x in h) else "DIFFERS FROM PRODUCTION")
except Exception as e: print("logprob missing", e)
arms = "${ARMS:-base exact conf}".split()
V = {}
for a in arms:
    try: V[a] = json.load(open(os.path.expanduser("~/vg-dc%s.json" % a)))
    except Exception: pass
import statistics
for a, d in V.items():
    print(f"{a:10s} c1 median {statistics.median(d['c1']):.2f} {d['c1']} tok_step {d.get('c1_tok_step')} c2 {d['c2']['aggregate']} c4 {d['c4']['aggregate']} short {d.get('short_wall_s')}")
if "base" in V:
    b = V["base"]
    for a, d in V.items():
        if a == "base": continue
        same = {k: (d["greedy"][k] == b["greedy"][k]) for k in b.get("greedy", {})}
        print(f"{a}: greedy text equal to base per prompt {same}; c1 ratio {statistics.median(d['c1'])/statistics.median(b['c1']):.3f} c2 {d['c2']['aggregate']/b['c2']['aggregate']:.3f} c4 {d['c4']['aggregate']/b['c4']['aggregate']:.3f}")
PY
echo DECODE-DONE
