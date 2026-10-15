#!/bin/bash
# ab_up.sh PRODUCT ARM IMAGE : both ranks through the product start-node.sh (pair must be held), fresh NVMe prefix-cache dir.
set -u
P=$1 ARM=$2 IMG=$3; W=RANK1_HOST; H=~
[ -d $H/sparkglm-gpu.lock ] || { echo "take the pair first"; exit 1; }
case $P in
  sparkglm)  MR=$H/sparkglm-cleanroom/models; EXTRA="--profile ${GLM_PROFILE:-$H/sparkglm-shared-20261014/install/profiles/4x1m.json} --gpu-memory-utilization 0.91 --display-carveout --kv-shard --prefix-cache-dir $H/shared-ab/pcd --prefix-cache-gb 48" ;;
  sparkqwen) MR=$H/.lloom/models;           EXTRA="--profile $H/sparkqwen-shared-20261014/install/profiles/8x262k.json --gpu-memory-utilization 0.70" ;;
esac
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n atlas-sparkqwen-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n atlas-sparkqwen-rank$n >/dev/null 2>&1"; done
rm -rf $H/shared-ab/pcd; mkdir -p $H/shared-ab/pcd; ssh -n $W "rm -rf $H/shared-ab/pcd; mkdir -p $H/shared-ab/pcd"
for i in $(seq 60); do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); b=$(ssh -n $W "awk \"/MemAvailable/{print int(\\\$2/1048576)}\" /proc/meminfo"); [ "$a" -ge 108 ] && [ "$b" -ge 108 ] && break; sleep 3; done
echo "== $P $ARM: MemAvailable $a / $b GiB; image $IMG"
ssh -n $W "cd $H/$P-shared-20261014 && bash install/start-node.sh --rank 1 --leader-address RANK0_HOST --model-root $MR --image $IMG --name atlas-sparkglm-rank1 --fabric-hca rocep1s0f0,roceP2p1s0f0 --fabric-interface enp1s0f0np0 $EXTRA" >/dev/null || exit 1
cd $H/$P-shared-20261014 && bash install/start-node.sh --rank 0 --leader-address RANK0_HOST --model-root $MR --image $IMG --name atlas-sparkglm-rank0 --fabric-hca rocep1s0f0,roceP2p1s0f0 --fabric-interface enp1s0f0np0 $EXTRA >/dev/null || exit 1
t0=$(date +%s)
for i in $(seq 240); do curl -sf http://127.0.0.1:8893/health >/dev/null && break
  if [ "$(docker inspect -f {{.State.Status}} atlas-sparkglm-rank0 2>/dev/null)" != running ] || [ "$(ssh -n $W docker inspect -f {{.State.Status}} atlas-sparkglm-rank1 2>/dev/null)" != running ]; then
    echo "== $ARM FAILED TO START"; docker logs --tail 15 atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-240; ssh -n $W "docker logs --tail 15 atlas-sparkglm-rank1 2>&1" | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-240; exit 1; fi
  sleep 5; done
curl -sf http://127.0.0.1:8893/health >/dev/null || { echo "== $ARM no health after 20 min"; exit 1; }
echo "== $ARM up in $(( $(date +%s)-t0 )) s: $(docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "[0-9]+ blocks × [0-9]+ tok/block = [0-9]+ max KV tokens|agree on [^,]*" | tr "\n" " ")"
echo "   engine: $(docker exec atlas-sparkglm-rank0 sh -c "cat /opt/atlas/source-manifest.json 2>/dev/null" | python3 -c "import json,sys; m=json.load(sys.stdin); print(m.get(\"branch\"), m.get(\"commit\",\"\")[:8])" 2>/dev/null)  MemAvailable $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) / $(ssh -n $W "awk \"/MemAvailable/{print int(\\\$2/1048576)}\" /proc/meminfo") GiB"
