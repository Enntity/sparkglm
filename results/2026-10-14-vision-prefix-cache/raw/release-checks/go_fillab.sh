#!/bin/bash
# go_fillab.sh NEW_IMAGE: fill-check A/B (production image vs NEW_IMAGE, same fresh seeds); derived from go_bench.sh, the SparkGLM release suite (go_final_bench.sh on 01, plus the 500+24 quality set, prose
# bursts and an arithmetic content probe) for NEW_IMAGE and the production image 5ec57d55, back to back on the
# 03/04 pair in production configuration. Outputs ~/final-bench/<label> and ~/field-final/<label>-<tokens> on 03.
NEW=${1:?usage: go_bench.sh NEW_IMAGE}
BASE=ghcr.io/enntity/atlas-sparkglm@sha256:5ec57d5522a95463b600a17471ce60712b315566e04c0a3bb6af1d84a1de0a4a
D=$HOME/sparkglm-dev; W=10.100.112.2; MR=$HOME/sparkglm-cleanroom/models; U=http://127.0.0.1:8893; M=glm-5.3-flash-atlas
[ -d $HOME/sparkglm-gpu.lock ] || { echo "take the 03/04 marker lock first"; exit 1; }
memwait() { for i in $(seq 60); do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); b=$(ssh -n $W "awk \"/MemAvailable/{print int(\\\$2/1048576)}\" /proc/meminfo")
    [ "$a" -ge 100 ] && [ "$b" -ge 100 ] && return 0; sleep 2; done; echo "memwait: only $a / $b GiB available"; }
run() {  # run TAG IMAGE
  docker rm -f atlas-sparkglm-rank0 >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank1 >/dev/null 2>&1"; memwait
  ssh $W "bash -s -- --rank 1 --leader-address 10.100.112.1 --model-root $MR --image $2 --profile $D/prod_4x1m.json" < $D/start-node-img.sh >/dev/null
  bash $D/start-node-img.sh --rank 0 --leader-address 10.100.112.1 --model-root $MR --image $2 --profile $D/prod_4x1m.json >/dev/null
  for i in $(seq 150); do curl -sf $U/health >/dev/null && break
    if [ "$(docker inspect -f {{.State.Status}} atlas-sparkglm-rank0 2>/dev/null)" != running ]; then echo "== $1 FAILED TO START"; exit 1; fi
    sleep 5; done
  echo "== $1 up ($2): $(docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "agree on [0-9]+|carveout not lent" | tr "\n" " ")"
}
fills() {  # fills TAG IMAGE: a fresh server, then fresh-seed 4x204K rounds 9001..9005
  run $1 $2; cd $D
  for s in 1 2 3 4 5; do FILL_SEED=$((9000 + s)) python3 pc_fill_fresh.py $1-fill$s 2>&1 | tail -1; done
  docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > $D/fab-$1.rank0.log
  grep -ahE "NVMe prefix restore" $D/fab-$1.rank0.log | tail -1 | sed "s/^.*INFO //" | cut -c1-200
}
fills fab-prod $BASE
fills fab-new $NEW
docker rm -f atlas-sparkglm-rank0 >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank1 >/dev/null 2>&1"
echo FILLAB-DONE
