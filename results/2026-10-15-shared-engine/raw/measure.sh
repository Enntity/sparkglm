#!/bin/bash
# measure.sh TAG IMAGE : fresh GLM start, then where rank 0's memory sits; teardown.
cd ~/shared-ab; T=$1; sleep 60
(while true; do echo "$(date -u +%T.%N | cut -c1-12) $(awk '/MemAvailable/{a=$2} /SUnreclaim/{s=$2} END{print a, s}' /proc/meminfo)"; sleep 0.5; done > memtl-$T.log) & S=$!
bash ab_up.sh sparkglm "$T" "$2" >/dev/null || { echo "$T failed to start"; kill $S; exit 1; }
sleep 30; kill $S
docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > full-$T.rank0.log
pid=$(docker top atlas-sparkglm-rank0 -eo pid,comm | awk '/spark/{print $1; exit}')
echo "== $T pid $pid"
grep -E 'VmRSS|RssAnon|RssFile|RssShmem|VmData|VmSize' /proc/$pid/status | tr -s ' \t' ' ' | tr '\n' ' '; echo
awk '/^(Rss|Pss|Anonymous|Shared_Clean|Private_Dirty)/' /proc/$pid/smaps_rollup | tr -s ' ' | tr '\n' ' '; echo
cp /proc/meminfo meminfo-$T.txt; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | head -3
echo "MemAvailable kB $(awk '/MemAvailable/{print $2}' /proc/meminfo)  Cached kB $(awk '/^Cached/{print $2}' /proc/meminfo)  Shmem kB $(awk '/^Shmem:/{print $2}' /proc/meminfo)"
# resident memory by mapping (from smaps), top entries
awk '/^[0-9a-f]+-[0-9a-f]+ /{name=($6==""?"[anon]":$6)} /^Rss:/{r[name]+=$2} END{for(k in r) if(r[k]>51200) printf "%8.0f MB %s\n", r[k]/1024, k}' /proc/$pid/smaps | sort -rn | head -12
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n RANK1_HOST "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
