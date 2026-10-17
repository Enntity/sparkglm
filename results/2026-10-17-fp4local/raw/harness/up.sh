#!/bin/bash
# up.sh ARM CKPT BIN PROFILE [UTIL]: both ranks through ~/glm-fp4l/install/start-node.sh (pair must be held).
# CKPT = fp4l | nvfp4 ; BIN = path to a spark binary or "image" ; PROFILE = JSON path or shipped name.
set -u
ARM=$1 CK=$2 BIN=$3 PROF=$4 UTIL=${5:-0.91}; W=RANK1_HOST; H=~; IMG=ghcr.io/enntity/atlas-sparkglm:f0f8a55c3da2
case $CK in
  fp4l)  MR=$H/.lloom/models; CK_ENV="CKPT_DIR=$H/.lloom/models/nvidia--GLM-5.3-Flash-FP4-Local" ;;
  nvfp4) MR=$H/sparkglm-cleanroom/models; CK_ENV="" ;;
  *) echo "CKPT fp4l|nvfp4"; exit 2 ;;
esac
BIN_ENV=""; [ "$BIN" != image ] && BIN_ENV="DEV_BIN=$BIN"
PC=$H/glm-fp4l/pcd
EXTRA="--profile $PROF --gpu-memory-utilization $UTIL --display-carveout --kv-shard --prefix-cache-dir $PC --prefix-cache-gb 48"
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n atlas-sparkqwen-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n atlas-sparkqwen-rank$n >/dev/null 2>&1"; done
rm -rf $PC; mkdir -p $PC; ssh -n $W "rm -rf $PC; mkdir -p $PC"
for i in $(seq 60); do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); b=$(ssh -n $W "awk \"/MemAvailable/{print int(\\\$2/1048576)}\" /proc/meminfo"); [ "$a" -ge 108 ] && [ "$b" -ge 108 ] && break; sleep 3; done
echo "== $ARM ($CK, $BIN, $(basename $PROF), util $UTIL): MemAvailable $a / $b GiB"
ssh -n $W "cd $H/glm-fp4l/install && env $CK_ENV $BIN_ENV bash start-node.sh --rank 1 --leader-address RANK0_HOST --model-root $MR --image $IMG --name atlas-sparkglm-rank1 --fabric-hca rocep1s0f0,roceP2p1s0f0 --fabric-interface enp1s0f0np0 $EXTRA" >/dev/null || exit 1
cd $H/glm-fp4l/install && env $CK_ENV $BIN_ENV bash start-node.sh --rank 0 --leader-address RANK0_HOST --model-root $MR --image $IMG --name atlas-sparkglm-rank0 --fabric-hca rocep1s0f0,roceP2p1s0f0 --fabric-interface enp1s0f0np0 $EXTRA >/dev/null || exit 1
t0=$(date +%s)
for i in $(seq 240); do curl -sf http://127.0.0.1:8893/health >/dev/null && break
  if [ "$(docker inspect -f {{.State.Status}} atlas-sparkglm-rank0 2>/dev/null)" != running ] || [ "$(ssh -n $W docker inspect -f {{.State.Status}} atlas-sparkglm-rank1 2>/dev/null)" != running ]; then
    echo "== $ARM FAILED TO START"; docker logs --tail 25 atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-300; ssh -n $W "docker logs --tail 25 atlas-sparkglm-rank1 2>&1" | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-300; exit 1; fi
  sleep 5; done
curl -sf http://127.0.0.1:8893/health >/dev/null || { echo "== $ARM no health after 20 min"; exit 1; }
echo "== $ARM up in $(( $(date +%s)-t0 )) s: $(docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "display carveout [0-9]+ MiB holds .* → [0-9]+ blocks|[0-9]+ blocks × [0-9]+ tok/block = [0-9]+ max KV tokens" | tail -2 | tr "\n" " ") (final pool = the carveout block count × 16 when the carveout is on)  MemAvailable $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) / $(ssh -n $W "awk \"/MemAvailable/{print int(\\\$2/1048576)}\" /proc/meminfo") GiB"
