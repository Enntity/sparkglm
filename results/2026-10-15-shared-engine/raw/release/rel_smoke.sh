#!/bin/bash
# rel_smoke.sh : smoke the release images on the pair (hold it first): GLM full suite + cold prefill + fill; Qwen hash + sq_bench, then decode A/B/A/B vs production.
set -u; cd ~/rel-ab; W=RANK1_HOST
GI=ghcr.io/enntity/atlas-sparkglm:948aabe47fa4; QI=ghcr.io/enntity/atlas-sparkqwen:9a0df5f1474a
QPROD=ghcr.io/enntity/atlas-sparkqwen@sha256:ac3a5c7da4841cacc810755da01b074b8216224409d7937a19dcfced569b72f0
down() { for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done; }
echo "##### start $(date -Is)"
echo "##### GLM $(date +%T)"; sleep 20
bash rel_up.sh sparkglm rel $GI && { timeout 3000 python3 prefill_probe.py rel 8192 32768 65536 131072; bash glm_suite.sh rel; }
down; sleep 45
bash rel_up.sh sparkglm relfill $GI && {
  docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -aE "KV budget self-relative|own-footprint|for KV|display carveout .* holds|KV cache: [0-9]+ blocks" | cut -c60-330 > kv-relfill.txt; cat kv-relfill.txt
  (while true; do echo "$(date +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 5; done > mem-fill-relfill.log) & S=$!
  timeout 1500 python3 ~/sparkglm-dev/pc_fill.py glm-relfill 2>&1 | tail -1; kill $S
  echo "   min MemAvailable rank0 kB during fill: $(sort -k2 -n mem-fill-relfill.log | head -1)"; }
down; sleep 45
echo "##### Qwen $(date +%T)"
bash rel_up.sh sparkqwen qrel $QI && bash qwen_smoke.sh qrel decode prefill conc warm; down; sleep 30
for a in qprod:$QPROD qrel2:$QI qprod2:$QPROD; do bash rel_up.sh sparkqwen ${a%%:*} ${a#*:} && bash qwen_smoke.sh ${a%%:*} decode; down; sleep 30; done
echo "##### done $(date -Is)"
