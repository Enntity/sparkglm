#!/bin/bash
# promo-tp2.sh: SparkGLM 0.3.6 promotion checks on the pair. Install through LLooM recipe version 7, then
# Spark-mode output against the 0.3.6 qualification, image and video output against 0.3.5, Strict output against 0.3.5.
set -u; export PATH=$HOME/.local/bin:$PATH; D=~/sg-036; Q=~/qual-20261103/kit; export SQ_MODEL=glm-5.3-flash-atlas
IMG=ghcr.io/enntity/atlas-sparkglm@sha256:e241d69dc3d1c592c6c605ad8e01b67763ff96178676282d40ccea18f43373d8
PEER=RANK1_FABRIC_ADDR; C=lloom-atlas-sparkglm-spark01
up() { # LABEL
  sleep 45
  for try in 1 2 3 4 5; do out=$(lloom resume glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error|error" | head -2); echo "$out"; echo "$out" | grep -qi error || break; sleep 45; done
  for i in $(seq 90); do curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas && break; sleep 5; done
  curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas || { echo "NOT SERVING $1"; docker logs --tail 15 $C 2>&1 | cut -c1-300; return 1; }
  docker inspect $C --format "{{.Config.Image}}"
  docker exec $C sh -c "tr \"\\0\" \"\\n\" < /proc/1/environ | grep -E \"^SPARKGLM_(MODE|PROFILE|GPU_MEMORY_UTILIZATION)\"" | tr "\n" " "; echo
  docker logs $C 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "KV cache: .*max KV|display carveout .* blocks" | cut -c1-230 | head -2
  echo "=== serving $1 $(date -u +%T)"
}
down() { lloom suspend glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error" | head -1; }
mode() { python3 ~/sg-035/mode-conf.py $IMG $1 && scp -q ~/sg-035/mode-conf.py $PEER:/tmp/mode-conf.py && ssh $PEER "python3 /tmp/mode-conf.py $IMG $1"; }

echo "=== INSTALL $(date -u +%T)"; down
scp -q $D/setup-v7.sh $PEER:sg-036/setup-v7.sh; ssh $PEER "bash ~/sg-036/setup-v7.sh 2 apply" | cut -c1-700
bash $D/setup-v7.sh 2 apply | cut -c1-700
ssh $PEER "docker images --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}}' | grep e241d69d | head -2"
up recipe-v7 || { echo "=== INSTALL FAILED"; exit 1; }

mkdir -p $D/spark $D/strict; cd $D/spark
cp ~/qual-20261103/sg-sg036-1118/greedy-sg036-1118.json ~/qual-20261103/sg-sg036-1118/lpd-sg036-1118.json .
python3 $Q/sq/greedy_eq.py run sg036-recipe > greedy.log 2>&1; echo "greedy rc $?"
python3 $Q/sq/lp_dump.py sg036-recipe > lpdump.log 2>&1; echo "lpdump rc $?"
python3 $Q/sq/greedy_eq.py compare sg036-1118 sg036-recipe > greedy-cmp.log 2>&1; python3 $Q/sq/lp_dump.py compare sg036-1118 sg036-recipe > lp-cmp.log 2>&1
tail -1 greedy-cmp.log | cut -c1-200; tail -4 lp-cmp.log | cut -c1-200
echo "=== SPARK DONE $(date -u +%T)"

cd ~/kv-reclaim && python3 vision_eq.py run sg036 > veq-sg036.log 2>&1; echo "vision run rc $?"; python3 vision_eq.py compare sg035 sg036 | tee $D/vision-cmp.log | tail -3
cp vision-sg036.json veq-sg036.log $D/ 2>/dev/null
docker logs $C 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "holds [0-9] buffers \([0-9]+ MiB\) → [0-9]+ blocks" | tail -1
echo "=== VISION DONE $(date -u +%T)"

down; mode strict; up strict || { echo "=== STRICT FAILED"; exit 1; }
cd $D/strict; cp ~/sg-035/strict/greedy-strict035.json ~/sg-035/strict/lpd-strict035.json .
python3 $Q/sq/greedy_eq.py run strict036 > greedy-strict036.log 2>&1; echo "greedy rc $?"
python3 $Q/sq/lp_dump.py strict036 > lpdump-strict036.log 2>&1; echo "lpdump rc $?"
python3 $Q/sq/greedy_eq.py compare strict035 strict036 > greedy-cmp.log 2>&1; python3 $Q/sq/lp_dump.py compare strict035 strict036 > lp-cmp.log 2>&1
tail -1 greedy-cmp.log | cut -c1-200; tail -4 lp-cmp.log | cut -c1-200
echo "=== STRICT DONE $(date -u +%T)"
down; mode spark
echo "=== TP2 CHECKS DONE $(date -u +%T)"
