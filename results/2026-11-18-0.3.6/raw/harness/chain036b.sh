#!/bin/bash
# chain036.sh: build the SparkGLM 0.3.6 candidate image from release/0.3.6, serve the product model on it, qualify it against
# the 0.3.5 receipts, run Sage's suite on it, then put the pair back on the 0.3.5 release configuration.
set -u; export PATH=$HOME/.local/bin:$PATH; peer=${WORKER:?ssh destination of the other Spark}; Q=~/qual-20261103/kit; M=glm-5.3-flash-atlas
SAGE=/home/enntitysparkadmin/sage/out/sage-c8; T=sg036-1118
resume() { for try in 1 2 3 4 5 6; do out=$(lloom resume $M --apply --yes 2>&1 | grep -E "applied|Error" | head -2); echo "$(date -u +%T) resume: $out"; echo "$out" | grep -q Error || break; sleep 45; done
  for i in $(seq 180); do curl -s -m 5 localhost:8893/v1/models | grep -q $M && break; sleep 5; done; curl -s -m 5 localhost:8893/v1/models | grep -q $M; }
shown() { docker exec lloom-atlas-sparkglm-ennspark01 sh -c 'tr "\0" "\n" < /proc/1/environ | grep -E "^(MODEL_PATH|SPARKGLM_MODE)"'; docker inspect lloom-atlas-sparkglm-ennspark01 --format '{{.Config.Image}}'; }
failed() { python3 - <<'PY'
import json
s = json.load(open("suite/summary.json")); bad = [(r["suite"], r["case"]) for r in s if not r["ok"]]
print(len(s), "cases,", len(bad), "failed:", bad)
PY
}
echo "=== wait for 0.3.6 $(date -u +%T)"
for i in $(seq 180); do curl -s -m 5 localhost:8893/v1/models | grep -q $M && break; sleep 5; done
curl -s -m 5 localhost:8893/v1/models | grep -q $M || { echo "0.3.6 NOT SERVING"; exit 1; }
shown; docker run --rm "$(docker inspect lloom-atlas-sparkglm-ennspark01 --format '{{.Config.Image}}')" --version 2>&1 | head -4
echo "=== start check $(date -u +%T)"; ~/thinkoff/lone.sh $T 2>&1 | tail -3
echo "=== qualify product $(date -u +%T)"
~/qual-20261103/run-sg.sh $T; cd ~/qual-20261103/sg-$T; cat exits | tr '\n' ' '; echo; failed; cut -c1-300 guardlog.log
cp ~/qual-20261103/sg-sg035-1114/greedy-sg035-1114.json ~/qual-20261103/sg-sg035-1114/lpd-sg035-1114.json .
python3 $Q/sq/greedy_eq.py compare $T sg035-1114 2>&1 | tail -3 | tee greedy-cmp.log
python3 $Q/sq/lp_dump.py compare $T sg035-1114 2>&1 | tail -4 | tee lp-cmp.log
grep -E "^(prose|code|structured) " speed.log; tail -5 matrix.log | cut -c1-200
echo "=== sage on 0.3.6 $(date -u +%T)"
lloom suspend $M --apply --yes 2>&1 | grep -E "applied|Error" | head -1
python3 ~/sage/sage-conf.py $SAGE && ssh -o BatchMode=yes $peer "python3 ~/sage/sage-conf.py $SAGE"; sleep 20
D=~/qual-20261103/sg-sage-c8-036-1118; mkdir -p "$D"
if resume; then shown; cd "$D"; python3 $Q/qualify/qualify.py --base-url http://127.0.0.1:8893 --output suite > suite.log 2>&1; echo "suite exit $?"; failed; else echo "SAGE NOT SERVING ON 0.3.6"; fi
echo "=== restore the 0.3.5 release configuration $(date -u +%T)"
lloom suspend $M --apply --yes 2>&1 | grep -E "applied|Error" | head -1
python3 ~/sage/sage-conf.py --restore ~/.lloom/config.before-sage-20261118T020129Z.json && ssh -o BatchMode=yes $peer "python3 ~/sage/sage-conf.py --restore ~/.lloom/config.before-sage-20261118T020130Z.json"
python3 ~/sage/pair-mem.py 0.91 114 && ssh -o BatchMode=yes $peer "python3 ~/sage/pair-mem.py 0.91 114"; sleep 30
out=$(lloom resume $M --apply --yes 2>&1 | grep -E "applied|Error" | head -1); echo "$(date -u +%T) resume at 0.91/114: $out"
if echo "$out" | grep -q Error; then  # 01 cannot offer 114 GB at admission: serve the release image with the 113 GB budget
  python3 ~/sage/pair-mem.py 0.90 113 && ssh -o BatchMode=yes $peer "python3 ~/sage/pair-mem.py 0.90 113"; sleep 10; echo "FELL BACK TO 0.90/113"
fi
if resume; then shown; docker exec lloom-atlas-sparkglm-ennspark01 sh -c 'tr "\0" "\n" < /proc/1/environ | grep UTILIZATION'; else echo "RELEASE CONFIG NOT SERVING"; fi
echo "CHAIN036 DONE $(date -u +%T)"
