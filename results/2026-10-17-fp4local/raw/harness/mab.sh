#!/bin/bash
# mab.sh TAG BIN [ENVFILE]: concurrency arm on 8x128k, kindling mixed4+mixed8 (series "mab"). PF_ARGS overrides argv. Run inside serve_lock.sh.
set -u; T=$1; BIN=$2; EF=${3:-}; G=~/glm-fp4l; D=$G/m-$T; mkdir -p $D; cd $D; W=RANK1_HOST
python3 - "$G/profiles/8x128k.json" "$D/8x128k.json" "$EF" "${PF_ARGS:-}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); ef = sys.argv[3]
if ef:
    for l in open(ef):
        l = l.strip()
        if l and not l.startswith("#"):
            k, v = l.split("=", 1); p["environment"][k] = v
for a in sys.argv[4].split():
    k = a.split("=", 1)[0]
    p["server_argv"] = [x for x in p["server_argv"] if x.split("=", 1)[0] != k] + [a]
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
ssh -n $W "mkdir -p $D"; rsync -a $D/8x128k.json $W:$D/
echo "##### $T mixed arm (env ${EF:-none}, args ${PF_ARGS:-none}) $(date +%T)"; [ -n "$EF" ] && grep -v "^#" $EF
bash $G/up.sh $T fp4l $BIN $D/8x128k.json 0.91 || { bash ~/fp4l/scan.sh $D M; exit 1; }
GATE_URL=http://127.0.0.1:8893 GATE_MODEL=glm-5.3-flash-atlas timeout 3000 python3 $G/kgate/bench/bench.py run --label $T --series mab --workloads mixed4,mixed8 --out $D/kb-$T.json 2>&1 | grep -E "^ *mixed" | head -4
bash ~/fp4l/scan.sh $D M
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
echo "##### $T mixed done $(date +%T)"
