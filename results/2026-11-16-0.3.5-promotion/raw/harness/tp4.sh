#!/bin/bash
# tp4.sh: SparkGLM 0.3.5 on four Sparks (recipe v6 --sparks 4) against the stored 0.3.0 TP-4 receipts (sg-tp4-030).
set -u; export PATH=$HOME/.local/bin:$PATH; Q=~/qual-20261103/kit; B=~/qual-20261103/sg-tp4-030; T=tp4-035
c=$(docker ps --format "{{.Names}}" | grep -m1 sparkglm); echo "container $c"
docker inspect $c --format "{{.Config.Image}}"
docker exec $c sh -c "tr \"\\0\" \"\\n\" < /proc/1/environ | grep -E \"^SPARKGLM_(MODE|PROFILE|GPU_MEMORY_UTILIZATION|KV_SHARD)|^NODE_RANK\""
docker logs $c 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "QUEUE_FIRST|OWNER_BATCH|LIVE_WIDTHS|KV cache: .*blocks|world|ranks" | cut -c1-220 | head -8
bash ~/qual-20261103/run-sg.sh $T $c > ~/sg-035/run-sg-$T.out 2>&1
cd ~/qual-20261103/sg-$T && cp $B/greedy-tp4-030.json $B/lpd-tp4-030.json . 2>/dev/null
export SQ_MODEL=glm-5.3-flash-atlas
python3 $Q/sq/greedy_eq.py compare tp4-030 $T > greedy-cmp.log 2>&1; python3 $Q/sq/lp_dump.py compare tp4-030 $T > lp-cmp.log 2>&1
cat exits | tr "\n" " "; echo; tail -1 greedy-cmp.log | cut -c1-120; tail -4 lp-cmp.log | cut -c1-160; tail -1 suite.log | cut -c1-120; grep "\"ok\": false" suite.log | cut -c1-220
grep -E "^(prose|code|structured) " speed.log | cut -c1-120; tail -5 matrix.log | cut -c1-200
cd ~/thinkoff && python3 verify.py 2>&1 | tail -1
echo "=== TP4 DONE $(date -u +%T)"
