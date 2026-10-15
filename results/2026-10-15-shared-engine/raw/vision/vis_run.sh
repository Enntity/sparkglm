#!/bin/bash
# vis_run.sh : on the held pair, test the vision-engine release images (SparkGLM suite + fill; SparkQwen vision
# gates, containment, cold-vs-warm caching, sq_bench) and run RigMark on them and on the production images.
set -u; cd ~/vis-ab; W=RANK1_HOST; R=~/rigmark
GN=ghcr.io/enntity/atlas-sparkglm:f0f8a55c3da2; QN=ghcr.io/enntity/atlas-sparkqwen:4e29c8c22671
GP=ghcr.io/enntity/atlas-sparkglm@sha256:94b837ef3ea7b3f54c158ce568e611405b1bd167edd56c734d8e90d2d6291cb6
QP=ghcr.io/enntity/atlas-sparkqwen@sha256:ac3a5c7da4841cacc810755da01b074b8216224409d7937a19dcfced569b72f0
EB='{"chat_template_kwargs":{"reasoning_effort":"low"}}'
sec() { echo "##### $* $(date +%T)"; }
down() { for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done; sleep 45; }
logs() { for c in 0 1; do if [ $c = 0 ]; then docker logs atlas-sparkglm-rank0 2>&1; else ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1"; fi | sed "s/\x1b\[[0-9;]*m//g" > $1.rank$c.log; done
  echo "-- $1 log scan: rank0 panics $(grep -ciE panicked $1.rank0.log) errors $(grep -cE " ERROR " $1.rank0.log); rank1 panics $(grep -ciE panicked $1.rank1.log) errors $(grep -cE " ERROR " $1.rank1.log)"; }
alive() { echo "-- containers: rank0 $(docker inspect -f {{.State.Status}} atlas-sparkglm-rank0 2>&1) rank1 $(ssh -n $W docker inspect -f {{.State.Status}} atlas-sparkglm-rank1 2>&1)"; }
rig_glm() {  # ARM ENGINE IMAGE: the RC4 release-suite RigMark cells (decode + prefill; concurrency + staggered)
  python3 meta.py $R/metadata.json $R/metadata-vis-$1.json "$2" "$3" 1048576
  (cd $R && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata-vis-$1.json --comparison-id vis-$1 --runs 3 --prefill-runs 2 \
    --prefill-depths 8192,32768,65536 --skip-concurrency --extra-body "$EB" --label $1-decode --output ~/vis-ab/rig-$1-decode.json > ~/vis-ab/rig-$1-decode.log 2>&1; echo "rig-decode exit $?"
   ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata-vis-$1.json --comparison-id vis-$1 --runs 1 --skip-prefill \
    --concurrency 1,2,3,4 --concurrency-runs 2 --extra-body "$EB" --staggered 2,4 --staggered-runs 2 --staggered-depth 32768 \
    --staggered-incumbent-tokens 1024 --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose \
    --label $1-conc --output ~/vis-ab/rig-$1-conc.json > ~/vis-ab/rig-$1-conc.log 2>&1; echo "rig-conc exit $?")
  sed -n '/R I G M A R K/,$p' rig-$1-decode.log rig-$1-conc.log | grep -E "GATES|CODE|PROSE|STRUCT|PREFILL|AGGREGATE|STAGGER" | head -16; }
rig_qwen() {  # ARM ENGINE IMAGE: the RC16 RigMark cells (8x32k at 0.88)
  python3 meta.py $R/metadata-sq.json $R/metadata-vis-$1.json "$2" "$3" 32768
  (cd $R && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata-vis-$1.json --comparison-id vis-$1 --runs 3 --prefill-runs 2 \
    --prefill-depths 16384 --concurrency 1,2,4,8 --concurrency-runs 1 --extra-body "$EB" --label $1 --output ~/vis-ab/rig-$1.json > ~/vis-ab/rig-$1.log 2>&1; echo "rigmark exit $?")
  sed -n '/R I G M A R K/,$p' rig-$1.log | grep -E "GATES|CODE|PROSE|STRUCT|PREFILL|AGGREGATE" | head -10; }

echo "##### start $(date -Is)"
sec "GLM new: cold prefill, suite, vision, RigMark"
bash vis_up.sh sparkglm gv $GN && {
  timeout 3000 python3 prefill_probe.py gv 8192 32768 65536 131072; bash glm_suite.sh gv
  (cd vpc-glm && python3 - <<'PY'
import json
for ph in ("race", "conv", "swap"):
    a = json.load(open(f"{ph}-gv.json")); b = json.load(open(f"{ph}-gfinal.json"))
    f = lambda d: [t["text"] for t in d["turns"]] if "turns" in d else {k: v["text"] for k, v in d["results"].items()}
    print(ph, "text identical to the pre-vision engine (gfinal):", f(a) == f(b))
PY
  )
  rig_glm gv "Atlas (Enntity/atlas sparkglm/atlas-20261015-vision @ 16315004), image $GN, SparkGLM 4x1m profile, GPU_MEMORY_UTILIZATION=0.91, display carveout, KV shard" "$GN via start-node.sh"
  logs glm-gv; }
down
sec "GLM new: fill"
bash vis_up.sh sparkglm gvfill $GN && {
  docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -aE "own-footprint|for KV|display carveout .* holds" | cut -c60-330 > kv-gvfill.txt; cat kv-gvfill.txt
  (while true; do echo "$(date +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 5; done > mem-fill-gvfill.log) & S=$!
  timeout 1500 python3 ~/sparkglm-dev/pc_fill.py glm-gvfill 2>&1 | tail -1; kill $S
  echo "   min MemAvailable rank0 kB during fill: $(sort -k2 -n mem-fill-gvfill.log | head -1)"; logs glm-gvfill; }
down
sec "GLM production: RigMark"
bash vis_up.sh sparkglm gp $GP && { rig_glm gp "Atlas (Enntity/atlas sparkglm/atlas-20261014-vpc @ 289cefbc), image ghcr.io/enntity/atlas-sparkglm:2b2970c2f05d, SparkGLM 4x1m profile, GPU_MEMORY_UTILIZATION=0.91, display carveout, KV shard" "ghcr.io/enntity/atlas-sparkglm:2b2970c2f05d via start-node.sh"; logs glm-gp; }
down
sec "Qwen new: hash, vision gates, sq_bench (8x262k, 0.70)"
bash vis_up.sh sparkqwen qv $QN && {
  echo "-- prompt-logprob hash (release 8713a659d5d3)"; timeout 600 python3 ~/sparkqwen-dev/lp_repeat.py qwen-qv 1000 2 2>&1 | tail -1; python3 -c "import json; print(\"hash\", [r[\"hash\"] for r in json.load(open(\"lp-qwen-qv.json\"))[\"runs\"]])"
  (cd vpc-qwen; for ph in race2 race conv swap; do echo "-- vpc $ph"; timeout 900 python3 vpc_gate.py qv $ph 2>&1 | tail -7; done)
  echo "-- sq_bench"; timeout 3600 python3 ~/sparkqwen-dev/sq_bench.py qwen-qv decode prefill conc warm 2>&1 | tail -30
  logs qwen-qv; grep -hE "Vision prefix cache:|mid-chunk SSM checkpoint saved" qwen-qv.rank0.log | cut -c30-220 | tail -6; }
down
sec "Qwen new: containment (vision markup escaping off)"
QWEN_PROFILE=$HOME/shared-ab/vfix-8x262k-markup0.json bash vis_up.sh sparkqwen qvm0 $QN && {
  (cd vpc-qwen; echo "-- vpc conv (expect a failed request, not a dead pair)"; timeout 900 python3 vpc_gate.py qvm0 conv 2>&1 | tail -4)
  alive; (cd vpc-qwen; echo "-- vpc race2 after the failure"; timeout 900 python3 vpc_gate.py qvm0 race2 2>&1 | tail -3)
  alive; logs qwen-qvm0; grep -hE "request failed on every rank" qwen-qvm0.rank*.log | cut -c1-200 | head -3; }
down
sec "Qwen new: cached turns equal cold ones (fresh servers)"
for n in 2 3 5; do bash vis_up.sh sparkqwen qvcold$n $QN >/dev/null && { echo "-- cold turn $n"; (cd vpc-qwen; timeout 900 python3 vpc_gate.py qvcold$n cold qv $n 2>&1 | tail -2); }; down; done
bash vis_up.sh sparkqwen qvsc $QN >/dev/null && { echo "-- swapcold"; (cd vpc-qwen; timeout 900 python3 vpc_gate.py qvsc swapcold qv 2>&1 | tail -4); logs qwen-qvsc; }
down
sec "Qwen new: RigMark (8x32k, 0.88)"
QWEN_PROFILE=$HOME/vis-sparkqwen/install/profiles/8x32k.json QWEN_UTIL=0.88 bash vis_up.sh sparkqwen qrv $QN && { rig_qwen qrv "Atlas (Enntity/atlas sparkqwen/atlas-20261015-vision @ 1954e759), image $QN, SparkQwen 8x32k profile, GPU_MEMORY_UTILIZATION=0.88" "$QN via start-node.sh"; logs qwen-qrv; }
down
sec "Qwen production: RigMark (8x32k, 0.88)"
QWEN_PROFILE=$HOME/vis-sparkqwen/install/profiles/8x32k.json QWEN_UTIL=0.88 bash vis_up.sh sparkqwen qrp $QP && { rig_qwen qrp "Atlas (Enntity/atlas sparkqwen/atlas-20261013-prefill @ fcf9735f), image ghcr.io/enntity/atlas-sparkqwen:a678dfce7c4c, SparkQwen 8x32k profile, GPU_MEMORY_UTILIZATION=0.88" "ghcr.io/enntity/atlas-sparkqwen:a678dfce7c4c via start-node.sh"; logs qwen-qrp; }
down
echo "##### done $(date -Is)"
