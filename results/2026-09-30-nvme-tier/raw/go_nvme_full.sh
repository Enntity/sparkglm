#!/bin/bash
# NVMe prefix tier through the installer (release/nvme-tier), full pool, 4x512k at 0.93: build the image from a fresh clone,
# start with PREFIX_CACHE_DIR/PREFIX_CACHE_GB, hash gate, vision check, then overflow the pool with ~1.84M tokens of conversations.
W=192.0.2.2; D=$HOME/sparkglm-dev
[ -d $HOME/sparkglm-gpu.lock ] || { echo "take the pair first"; exit 1; }
cd ~/sparkglm-pc && rm -rf sparkglm-nvme && git clone -q -b release/nvme-tier https://github.com/Enntity/sparkglm.git sparkglm-nvme && cd sparkglm-nvme
printf "WORKER=$W\nMODEL_ROOT=%s\nBUILD=1\nGPU_MEMORY_UTILIZATION=0.93\nPREFIX_CACHE_DIR=%s\nPREFIX_CACHE_GB=48\n" "$HOME/sparkglm-cleanroom/models" "$HOME/prefix-cache" > .env
echo "== $(git log --oneline -1) $(install/build.sh --tag)"
t0=$(date +%s); ./start.sh 2>&1 | grep -E "^.{0,12}== |smoke|refus|error|Error" | sed "s/\x1b\[[0-9;]*m//g"; echo "== start.sh took $(( $(date +%s) - t0 )) s"
for r in 0 1; do echo "-- rank $r"; if [ $r = 0 ]; then docker logs atlas-sparkglm-rank0 2>&1; else ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1"; fi | sed "s/\x1b\[[0-9;]*m//g" | grep -aiE "KV cache: .*max KV|agree on|nvme|swap file|reserve" | cut -c40-300 | head -8; done
docker inspect -f '{{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}' atlas-sparkglm-rank0
(while true; do echo "$(date +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 5; done > $D/nvfull-mem0.log) & s0=$!
cd $D
echo "== logprob hash"; timeout 300 python3 lp_repeat.py nvfull 2100 1 2>&1 | tail -1
echo "== vision"; timeout 300 python3 vision_check.py 2>&1 | tail -3
echo "== full-pool eviction"; timeout 2400 python3 evict_full.py nvfull 10 8000 2>&1 | tail -16
kill $s0
for r in 0 1; do if [ $r = 0 ]; then docker logs atlas-sparkglm-rank0 2>&1; else ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1"; fi | sed "s/\x1b\[[0-9;]*m//g" > $D/nvfull.rank$r.log
  echo "rank$r: errors $(grep -acE "panicked|ERROR " $D/nvfull.rank$r.log); restores $(grep -aci "nvme.*restor" $D/nvfull.rank$r.log); spills $(grep -aci "nvme.*spill\|spilled" $D/nvfull.rank$r.log)"; done
grep -aiE "nvme.*restor" $D/nvfull.rank0.log | tail -4 | cut -c40-300
echo "min MemAvailable kB rank0 $(sort -k2 -n $D/nvfull-mem0.log | head -1)"; du -sh $HOME/prefix-cache/* 2>/dev/null
echo NVFULL-DONE
