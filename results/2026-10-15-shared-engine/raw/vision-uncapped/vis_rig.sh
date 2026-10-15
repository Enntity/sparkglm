#!/bin/bash
# vis_rig.sh : RigMark on the held pair, GPU clocks per CLOCKS: uncapped (default) stops gpu-clock-cap.service on both
# Sparks for the run (nvidia-smi -rgc) and an exit trap starts it again; capped leaves the cap on.
# SparkGLM new (release-suite cells), SparkQwen new and production (RC16 cells, 8x32k at 0.88).
set -u; cd ~/vis-ab; W=RANK1_HOST; R=~/rigmark; CLOCKS=${CLOCKS:-uncapped}; S=${SUFFIX:-u}
GN=ghcr.io/enntity/atlas-sparkglm:f0f8a55c3da2; QN=ghcr.io/enntity/atlas-sparkqwen:4e29c8c22671
QP=ghcr.io/enntity/atlas-sparkqwen@sha256:ac3a5c7da4841cacc810755da01b074b8216224409d7937a19dcfced569b72f0
EB='{"chat_template_kwargs":{"reasoning_effort":"low"}}'
clocks() { sudo -n systemctl $1 gpu-clock-cap.service && ssh -n $W "sudo -n systemctl $1 gpu-clock-cap.service"; }
clock_state() { echo "-- clocks ($1): cap 01 $(systemctl is-active gpu-clock-cap.service) 02 $(ssh -n $W systemctl is-active gpu-clock-cap.service); 01 $(nvidia-smi --query-gpu=clocks.max.graphics,clocks.applications.graphics --format=csv,noheader)"; }
if [ "$CLOCKS" = uncapped ]; then
  clocks stop || { echo "cannot lift the clock cap (sudoers rule missing?)"; exit 1; }
  trap 'clocks start; clock_state restored' EXIT; trap 'exit 1' INT TERM
fi
clock_state "$CLOCKS"
sec() { echo "##### $* $(date +%T)"; }
down() { for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done; sleep 45; }
logs() { for c in 0 1; do if [ $c = 0 ]; then docker logs atlas-sparkglm-rank0 2>&1; else ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1"; fi | sed "s/\x1b\[[0-9;]*m//g" > $1.rank$c.log; done
  echo "-- $1 log scan: rank0 panics $(grep -ciE panicked $1.rank0.log) errors $(grep -cE " ERROR " $1.rank0.log); rank1 panics $(grep -ciE panicked $1.rank1.log) errors $(grep -cE " ERROR " $1.rank1.log)"; }
alive() { echo "-- containers: rank0 $(docker inspect -f {{.State.Status}} atlas-sparkglm-rank0 2>&1) rank1 $(ssh -n $W docker inspect -f {{.State.Status}} atlas-sparkglm-rank1 2>&1)"; }
rig_glm() {  # ARM ENGINE IMAGE: the RC4 release-suite RigMark cells (decode + prefill; concurrency + staggered)
  python3 meta.py $R/metadata.json $R/metadata-vis-$1.json "$2" "$3" 1048576
  (cd $R && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata-vis-$1.json --comparison-id vis-$1 --runs 3 --prefill-runs 2 \
    --prefill-depths 8192,32768,65536 --skip-concurrency --extra-body "$EB" --label $1-decode --output ~/vis-ab/rig-$1-decode.json > ~/vis-ab/rig-$1-decode.log 2>&1; echo "rig-decode exit $?"
   ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata-vis-$1.json --comparison-id vis-$1 --runs 1 --skip-prefill \
    --concurrency 1,2,3,4 --concurrency-runs 2 --extra-body "$EB" --staggered 2,4 --staggered-runs 2 --staggered-depth 32768 \
    --staggered-incumbent-tokens 1024 --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose \
    --label $1-conc --output ~/vis-ab/rig-$1-conc.json > ~/vis-ab/rig-$1-conc.log 2>&1; echo "rig-conc exit $?")
  sed -n '/R I G M A R K/,$p' rig-$1-decode.log rig-$1-conc.log | grep -E "GATES|CODE|PROSE|STRUCT|PREFILL|AGGREGATE|STAGGER" | head -16; }
rig_qwen() {  # ARM ENGINE IMAGE: the RC16 RigMark cells (8x32k at 0.88)
  python3 meta.py $R/metadata-sq.json $R/metadata-vis-$1.json "$2" "$3" 32768
  (cd $R && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata-vis-$1.json --comparison-id vis-$1 --runs 3 --prefill-runs 2 \
    --prefill-depths 16384 --concurrency 1,2,4,8 --concurrency-runs 1 --extra-body "$EB" --label $1 --output ~/vis-ab/rig-$1.json > ~/vis-ab/rig-$1.log 2>&1; echo "rigmark exit $?")
  sed -n '/R I G M A R K/,$p' rig-$1.log | grep -E "GATES|CODE|PROSE|STRUCT|PREFILL|AGGREGATE" | head -10; }

echo "##### start $(date -Is) CLOCKS=$CLOCKS"
sec "GLM new: RigMark"
bash vis_up.sh sparkglm gv$S $GN && { rig_glm gv$S "Atlas (Enntity/atlas sparkglm/atlas-20261015-vision @ 16315004), image $GN, SparkGLM 4x1m profile, GPU_MEMORY_UTILIZATION=0.91, display carveout, KV shard, GPU clocks $CLOCKS" "$GN via start-node.sh"; logs glm-gv$S; }
down
qwen_arm() {  # ARM IMAGE ENGINE-DESCRIPTION PUBLISHED-IMAGE
  sec "Qwen $1: RigMark (8x32k, 0.88)"
  QWEN_PROFILE=$HOME/vis-sparkqwen/install/profiles/8x32k.json QWEN_UTIL=0.88 bash vis_up.sh sparkqwen $1 $2 && {
    rig_qwen $1 "$3, SparkQwen 8x32k profile, GPU_MEMORY_UTILIZATION=0.88, GPU clocks $CLOCKS" "$4 via start-node.sh"; logs qwen-$1; }
  down; }
qwen_arm qrv$S $QN "Atlas (Enntity/atlas sparkqwen/atlas-20261015-vision @ 1954e759), image $QN" $QN
qwen_arm qrp$S $QP "Atlas (Enntity/atlas sparkqwen/atlas-20261013-prefill @ fcf9735f), image ghcr.io/enntity/atlas-sparkqwen:a678dfce7c4c" ghcr.io/enntity/atlas-sparkqwen:a678dfce7c4c
clock_state "end of run"
echo "##### done $(date -Is)"
