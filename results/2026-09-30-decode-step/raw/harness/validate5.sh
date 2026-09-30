#!/bin/bash
# Release validation: build the release branch from a fresh clone (image + engine), start it with ./start.sh, and run the
# full harness on the image as shipped. Pair must be held. BR = branch (default release/exact-speedups).
set -e
BR=${BR:-release/exact-speedups}; D=$HOME/sparkglm-dev; W=192.0.2.2; OUT=$HOME/sparkglm-pc/out5
[ -d $HOME/sparkglm-gpu.lock ] || { echo "take the pair first"; exit 1; }
cd ~/sparkglm-pc/sparkglm && ./start.sh stop >/dev/null 2>&1 || true
cd ~/sparkglm-pc && rm -rf sparkglm $OUT && git clone -q -b $BR https://github.com/Enntity/sparkglm.git && cd sparkglm && mkdir -p $OUT
printf "WORKER=$W\nMODEL_ROOT=%s\nBUILD=1\nGPU_MEMORY_UTILIZATION=0.93\n" "$HOME/sparkglm-cleanroom/models" > .env
echo "== $(git log --oneline -1) $(install/build.sh --tag)"
t0=$(date +%s); ./start.sh 2>&1 | grep -E "^.{0,12}== |smoke" | sed "s/\x1b\[[0-9;]*m//g"; echo "== start.sh took $(( $(date +%s) - t0 )) s"
set +e
(while true; do echo "$(date +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 5; done > $OUT/memavailable-rank0.log) & s0=$!
(ssh $W "while true; do echo \"\$(date +%T) \$(awk \"/MemAvailable/{print \\\$2}\" /proc/meminfo)\"; sleep 5; done" > $OUT/memavailable-rank1.log) & s1=$!
trap "kill $s0 $s1 2>/dev/null" EXIT
for r in 0 1; do echo "-- rank $r sizing"; if [ $r = 0 ]; then docker logs atlas-sparkglm-rank0 2>&1; else ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1"; fi | sed "s/\x1b\[[0-9;]*m//g" | grep -aE "KV budget self|KV own-footprint|KV cache: .*max KV|KV pool limited|agree on|Marconi [0-9]+ slots" | cut -c60-330; done
docker inspect -f "{{.Config.Image}}" atlas-sparkglm-rank0; docker exec atlas-sparkglm-rank0 python3 -c "import json; m=json.load(open(\"/opt/atlas/source-manifest.json\")); print(\"engine\", m[\"branch\"], m[\"commit\"][:8])"
cd $D
echo "== logprob hash"; timeout 600 python3 lp_repeat.py v5 2100 2 2>&1 | tail -3
echo "== needles"; timeout 600 python3 kvs_test.py v5 2>&1 | tail -4 | cut -c1-150
echo "== pc_test"; python3 pc_test.py v5 2>&1 | tail -9
echo "== pc_hard"; python3 pc_hard.py v5 2>&1 | grep -vE "^\s|Traceback|File " | tail -26
echo "== acc"; python3 pc_acc.py v5 2>&1 | tail -1
echo "== pc_policy"; timeout 600 python3 pc_policy_bench.py v5 2>&1 | tail -1 | cut -c1-600
echo "== decode"; timeout 600 python3 vg-probe-nx.py v5 2>&1 | tail -1 | cut -c1-300
echo "== fill"; python3 pc_fill.py v5 2>&1 | tail -14
cd ~/sparkglm-pc/sparkglm/bench
echo "== matrix"; MATRIX_SALT=mb-vx2-7c31 python3 run_matrix.py $OUT/out-matrix-1 matrix
echo "== field"; MATRIX_SALT=fg-$(date +%s) python3 run_matrix.py $OUT/out-field field
for r in 0 1; do if [ $r = 0 ]; then docker logs atlas-sparkglm-rank0 2>&1; else ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1"; fi | sed "s/\x1b\[[0-9;]*m//g" > $D/v5.rank$r.log; echo "rank$r error lines: $(grep -acE "panicked|ERROR " $D/v5.rank$r.log)"; done
echo "min MemAvailable kB rank0 $(sort -k2 -n $OUT/memavailable-rank0.log | head -1) rank1 $(sort -k2 -n $OUT/memavailable-rank1.log | head -1)"
kill $s0 $s1 2>/dev/null
echo "== 8x128k profile"
cd ~/sparkglm-pc/sparkglm && ./start.sh stop >/dev/null 2>&1; sleep 20
PROFILE=8x128k ./start.sh 2>&1 | grep -E "^.{0,12}== |smoke" | sed "s/\x1b\[[0-9;]*m//g"
docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -aE "KV cache: .*max KV|agree on" | cut -c60-300
cd $D && timeout 600 python3 kvs_test.py v5-8x 2>&1 | head -3 | cut -c1-150
cd ~/sparkglm-pc/sparkglm/bench && MATRIX_SALT=mb-8x-$(date +%s) python3 run_matrix.py $OUT/out-matrix-8x matrix c4-16k 2>&1 | tail -2
python3 $D/mm_streams.py 2>&1 | tail -4 || true
docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > $D/v5-8x.rank0.log; echo "8x rank0 error lines: $(grep -acE "panicked|ERROR " $D/v5-8x.rank0.log)"
cd ~/sparkglm-pc/sparkglm && ./start.sh stop >/dev/null 2>&1
echo VALIDATE-DONE
