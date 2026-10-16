#!/bin/bash
# glm_final.sh TAG IMAGE : fresh start, cold-prefill sweep, full GLM suite, teardown; then fresh start + fill.
cd ~/shared-ab; T=$1; I=$2
sleep 45; bash ab_up.sh sparkglm $T $I && { timeout 3000 python3 prefill_probe.py $T 8192 32768 65536 131072; bash glm_suite.sh $T; }
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n RANK1_HOST "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
bash fill_only.sh ${T}fill $I
echo "##### final done $(date -Is)"
