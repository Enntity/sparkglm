#!/bin/bash
# tebarm.sh TAG CKPT BIN ENVFILE|- : ToolEvalBench (default suite, seed 42, 3 trials, 4 parallel) on the production 4x1m profile (+ENVFILE). Run inside serve_lock.sh.
set -u; T=$1; CK=$2; BIN=$3; EF=$4; G=~/glm-fp4l; D=$G/t-$T; mkdir -p $D; cd $D; W=RANK1_HOST
python3 - "$G/profiles/4x1m.json" "$D/4x1m.json" "$EF" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
if sys.argv[3] != "-":
    for l in open(sys.argv[3]):
        l = l.strip()
        if l and not l.startswith("#"):
            k, v = l.split("=", 1); p["environment"][k] = v
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
ssh -n $W "mkdir -p $D"; rsync -a $D/4x1m.json $W:$D/
echo "##### $T ToolEvalBench ($CK, env $EF) $(date +%T)"
bash $G/up.sh $T $CK $BIN $D/4x1m.json 0.91 || { bash ~/fp4l/scan.sh $D T; exit 1; }
[ -f $G/tools_rfmt_probe.py ] && { for t in "" --thinking; do timeout 900 python3 $G/tools_rfmt_probe.py --url http://127.0.0.1:8893 --runs 5 $t 2>&1 | tail -2; done; grep -a -m1 "OR response_format" $D/../t-$T/T.rank0.log 2>/dev/null; }
timeout 10800 $G/teb-venv/bin/tool-eval-bench run --base-url http://127.0.0.1:8893 --model glm-5.3-flash-atlas --seed 42 --trials ${TRIALS:-3} --parallel 4 > $D/teb.log 2>&1; echo "teb exit $?"
grep -aiE "score|overall|pass|partial|fail" $D/teb.log | tail -12
bash ~/fp4l/scan.sh $D T
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
echo "##### $T done $(date +%T)"
