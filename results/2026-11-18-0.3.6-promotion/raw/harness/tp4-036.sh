#!/bin/bash
# tp4-036.sh: SparkGLM 0.3.6 on four Sparks (recipe version 7, --sparks 4) against the stored 0.3.5 four-Spark receipts (sg-tp4-035).
set -u; export PATH=$HOME/.local/bin:$PATH; Q=~/qual-20261103/kit; B=~/qual-20261103/sg-tp4-035; T=tp4-036
for try in 1 2 3 4 5; do out=$(lloom resume glm-5.3-flash-atlas --apply --yes 2>&1 | grep -iE "applied|error|short|cannot" | head -3); echo "$out"; echo "$out" | grep -qi "error\|cannot" || break; sleep 45; done
for i in $(seq 120); do curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas && break; sleep 5; done
c=$(docker ps --format "{{.Names}}" | grep -m1 sparkglm); echo "container $c"
curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas || { echo "NOT SERVING"; docker logs --tail 20 $c 2>&1 | cut -c1-300; echo "=== TP4 FAILED"; exit 1; }
docker inspect $c --format "{{.Config.Image}}"
docker exec $c sh -c "tr \"\\0\" \"\\n\" < /proc/1/environ | grep -E \"^SPARKGLM_(MODE|PROFILE|GPU_MEMORY_UTILIZATION|KV_SHARD)|^NODE_RANK\"" | tr "\n" " "; echo
docker logs $c 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > ~/sg-036/tp4-engine-start.log
grep -E "KV cache: .*blocks|DFlash rank split|4 ranks" ~/sg-036/tp4-engine-start.log | cut -c1-220 | head -6
echo "=== serving tp4 $(date -u +%T)"
bash ~/qual-20261103/run-sg.sh $T $c > ~/sg-036/run-sg-$T.out 2>&1
cd ~/qual-20261103/sg-$T && cp $B/greedy-tp4-035.json $B/lpd-tp4-035.json . 2>/dev/null
export SQ_MODEL=glm-5.3-flash-atlas
python3 $Q/sq/greedy_eq.py compare tp4-035 $T > greedy-cmp.log 2>&1; python3 $Q/sq/lp_dump.py compare tp4-035 $T > lp-cmp.log 2>&1
cat exits | tr "\n" " "; echo; tail -1 greedy-cmp.log | cut -c1-120; tail -4 lp-cmp.log | cut -c1-160; tail -1 suite.log | cut -c1-120; grep "\"ok\": false" suite.log | cut -c1-220
grep -E "^(prose|code|structured) " speed.log | cut -c1-120; tail -5 matrix.log | cut -c1-200
echo "--- 0.3.5 baseline"; grep -E "^(prose|code|structured) " $B/speed.log | cut -c1-120; tail -5 $B/matrix.log | cut -c1-200; tail -1 $B/suite.log | cut -c1-120
echo "=== TP4 DONE $(date -u +%T)"
lloom suspend glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error" | head -1
echo "=== TP4 DOWN $(date -u +%T)"
