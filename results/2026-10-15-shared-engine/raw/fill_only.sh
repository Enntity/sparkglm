#!/bin/bash
# fill_only.sh TAG IMAGE : fresh GLM start, four 204K sessions, rank-0 MemAvailable + container RSS sampler, teardown.
cd ~/shared-ab; T=$1
echo "##### start $(date -Is)"
sleep 60; bash ab_up.sh sparkglm $T $2 && {
  docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -aE "KV budget self-relative|own-footprint|for KV|display carveout .* holds|KV cache: [0-9]+ blocks|Sparse index cache" | cut -c60-330 > kv-$T.txt; cat kv-$T.txt;
  (while true; do echo "$(date +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo) $(docker stats --no-stream --format "{{.MemUsage}}" atlas-sparkglm-rank0 2>/dev/null | cut -d/ -f1)"; sleep 5; done > mem-fill-$T.log) & S=$!
  echo "idle: $(tail -1 mem-fill-$T.log)"; timeout 1500 python3 ~/sparkglm-dev/pc_fill.py glm-$T 2>&1 | tail -1; kill $S
  echo "   min MemAvailable rank0 kB during fill: $(sort -k2 -n mem-fill-$T.log | head -1)"; }
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n RANK1_HOST "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
echo "##### done $(date -Is)"
