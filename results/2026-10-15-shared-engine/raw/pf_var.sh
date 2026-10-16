#!/bin/bash
# pf_var.sh TAG IMAGE [PROFILE] : fresh start, cold prefill at 8K/32K x2, teardown.
cd ~/shared-ab; sleep 45
GLM_PROFILE=$3 bash ab_up.sh sparkglm $1 $2 && timeout 1200 python3 prefill_probe.py $1 ${SIZES:-8192 32768}; docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > full-$1.rank0.log
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n RANK1_HOST "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
