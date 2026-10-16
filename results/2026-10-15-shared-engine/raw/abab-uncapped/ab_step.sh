#!/bin/bash
# ab_step.sh ROUND : uncapped, alternating production and new (ABAB) per product, equal-text decode probes plus RigMark
# short-code concurrency. GLM: multi_probe (12 prose + 12 code x 256 forced tokens; tok/step and step time) + RigMark
# C1-C4. Qwen (8x32k, 0.88): sq_bench decode + RigMark C1/C2/C4/C8. Arms: ARMS env (default "gp gn gp gn qp qn qp qn").
set -u; cd ~/vis-ab; W=RANK1_HOST; R=~/rigmark; RD=${1:-r1}; CLOCKS=${CLOCKS:-uncapped}
GN=${GN:-ghcr.io/enntity/atlas-sparkglm:f0f8a55c3da2}; QN=${QN:-ghcr.io/enntity/atlas-sparkqwen:4e29c8c22671}
GP=ghcr.io/enntity/atlas-sparkglm@sha256:94b837ef3ea7b3f54c158ce568e611405b1bd167edd56c734d8e90d2d6291cb6
QP=ghcr.io/enntity/atlas-sparkqwen@sha256:ac3a5c7da4841cacc810755da01b074b8216224409d7937a19dcfced569b72f0
EB='{"chat_template_kwargs":{"reasoning_effort":"low"}}'
clocks() { sudo -n systemctl $1 gpu-clock-cap.service && ssh -n $W "sudo -n systemctl $1 gpu-clock-cap.service"; }
if [ "$CLOCKS" = uncapped ]; then clocks stop || { echo "cannot lift the cap"; exit 1; }; trap 'clocks start; echo "-- cap restored: $(systemctl is-active gpu-clock-cap.service)"' EXIT; trap 'exit 1' INT TERM; fi
echo "-- clocks $CLOCKS: cap 01 $(systemctl is-active gpu-clock-cap.service) 02 $(ssh -n $W systemctl is-active gpu-clock-cap.service)"
down() { for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done; sleep 40; }
conc() {  # LABEL LEVELS
  (cd $R && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json --comparison-id step-$1 --runs 1 --skip-prefill \
    --concurrency $2 --concurrency-runs 2 --extra-body "$EB" --label $1 --output ~/vis-ab/ab-$1.json > ~/vis-ab/ab-$1.log 2>&1)
  python3 -c "import json; d=json.load(open('ab-$1.json')); print('   conc', {k: round(v['aggregate_end_to_end_tokens_per_second']['median'],1) for k,v in d['concurrency'].items()})"; }
n=0
for a in ${ARMS:-gp gn gp gn qp qn qp qn}; do n=$((n+1)); T=$RD-$a$n
  case $a in
    gp|gn) I=$GP; [ $a = gn ] && I=$GN; echo "##### $T $(date +%T) $I"
      bash vis_up.sh sparkglm x$T $I >/dev/null && {
        python3 ~/sparkglm-dev/multi_probe.py $T | python3 -c "import json,sys; l=sys.stdin.read().split(' ',1); d=json.loads(l[1]); print('   multi', {k:(v['tok_s'], v['tok_step_mean'], round(1000*v['tok_step_mean']/v['tok_s'],2)) for k,v in d.items()}, '(tok/s, tok/step, ms/step)')"
        conc $T 1,2,3,4; } ;;
    qp|qn) I=$QP; [ $a = qn ] && I=$QN; echo "##### $T $(date +%T) $I"
      QWEN_PROFILE=$HOME/vis-sparkqwen/install/profiles/8x32k.json QWEN_UTIL=0.88 bash vis_up.sh sparkqwen x$T $I >/dev/null && {
        timeout 1200 python3 ~/sparkqwen-dev/sq_bench.py qwen-$T decode >/dev/null 2>&1; python3 -c "
import json,statistics as s; d=json.load(open('$HOME/sparkqwen-dev/bench-qwen-$T.json'))['cells']['decode']
print('   decode', {k: round(s.median([r['tok_s'] for r in d if r['kind']==k]),2) for k in ('prose','code','json')})"
        conc $T 1,2,4,8; } ;;
  esac
  down
done
echo "##### done $(date -Is)"
