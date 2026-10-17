#!/bin/bash
# bench.sh TAG CKPT BIN [ENVFILE]: speed arms (pair held, run inside serve_lock.sh, clocks uncapped).
#   P: 4x512k production profile -> kindling prefill.py 32k/128k cold + RigMark (code; prose; prefill 8K/32K/64K)
#   K: 8x128k -> kindling gate/bench/bench.py run (code/prose/structured/counting/mixed4/mixed8, thinking off)
# ENVFILE (optional): extra ATLAS_* lines merged into both profiles (A/B levers).
set -u; T=$1; CK=$2; BIN=$3; EF=${4:-}; G=~/glm-fp4l; D=$G/b-$T; mkdir -p $D; cd $D; W=RANK1_HOST
mkprof() { python3 - "$G/profiles/$1.json" "$D/$1.json" "$EF" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); ef = sys.argv[3]
if ef:
    for l in open(ef):
        l = l.strip()
        if l and not l.startswith("#"):
            k, v = l.split("=", 1); p["environment"][k] = v
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
rsync -a $D/$1.json $W:$D/ 2>/dev/null || { ssh -n $W "mkdir -p $D"; rsync -a $D/$1.json $W:$D/; }; }
mkprof 4x512k; mkprof 8x128k
EB='{"chat_template_kwargs":{"reasoning_effort":"low"}}'; R=~/rigmark
echo "##### $T arm P ($CK) $(date +%T)"
bash $G/up.sh $T-P $CK $BIN $D/4x512k.json 0.91 || { bash ~/fp4l/scan.sh $D P; exit 1; }
echo "-- kindling prefill (cold, random words)"; timeout 1800 python3 $G/kgate/prefill.py --base http://127.0.0.1:8893/v1 --model glm-5.3-flash-atlas --sizes 32k,128k --skip-thinking --out-tokens 64 2>&1 | tail -8
python3 ~/vis-ab/meta.py $R/metadata-sq.json $R/metadata-$T.json "sparkglm-fp4l $(basename $BIN) $CK" ghcr.io/enntity/atlas-sparkglm:f0f8a55c3da2 524288
cd $R && timeout 5400 ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata-$T.json --comparison-id glm-fp4l --runs 2 --prefill-runs 2 \
  --prefill-depths 8192,32768,65536 --concurrency 1,2,4 --concurrency-runs 1 --extra-body "$EB" --label $T --output $D/rig-$T.json > $D/rig-$T.log 2>&1; echo "rigmark exit $?"; cd $D
sed -n "/R I G M A R K/,\$p" $D/rig-$T.log | grep -E "GATES|CODE|PROSE|STRUCT|PREFILL|AGGREGATE|C[1248] |8K|32K|64K" | head -20
bash ~/fp4l/scan.sh $D P
echo "##### $T arm K ($CK, 8x128k) $(date +%T)"
bash $G/up.sh $T-K $CK $BIN $D/8x128k.json 0.91 || { bash ~/fp4l/scan.sh $D K; exit 1; }
GATE_URL=http://127.0.0.1:8893 GATE_MODEL=glm-5.3-flash-atlas timeout 5400 python3 $G/kgate/bench/bench.py run --label $T --out $D/kb-$T.json 2>&1 | tail -25
bash ~/fp4l/scan.sh $D K
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
echo "##### $T bench done $(date -Is)"
