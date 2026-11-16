#!/bin/bash
# strict.sh: Strict-mode output of 0.3.5 (published image, by digest) against the stored 0.3.3 Strict receipts.
set -u; export PATH=$HOME/.local/bin:$PATH; D=~/sg-035; Q=~/qual-20261103/kit; export SQ_MODEL=glm-5.3-flash-atlas
IMG=ghcr.io/enntity/atlas-sparkglm@sha256:ab56ad3e0879da5b039903fede4838d762f70d991509f2ad3598baca2446b872
boot() { # MODE
  lloom suspend glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error" | head -1
  python3 $D/mode-conf.py $IMG $1 && scp -q $D/mode-conf.py RANK1_FABRIC_ADDR:/tmp/mode-conf.py && ssh RANK1_FABRIC_ADDR "python3 /tmp/mode-conf.py $IMG $1"
  sleep 45
  for try in 1 2 3 4 5; do out=$(lloom resume glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error" | head -2); echo "$out"; echo "$out" | grep -q Error || break; sleep 45; done
  for i in $(seq 90); do curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas && break; sleep 5; done
  curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas || { echo "NOT SERVING $1"; docker logs --tail 15 lloom-atlas-sparkglm-ennspark01 2>&1 | cut -c1-300; return 1; }
  docker inspect lloom-atlas-sparkglm-ennspark01 --format "{{.Config.Image}}"
  docker exec lloom-atlas-sparkglm-ennspark01 sh -c "tr \"\\0\" \"\\n\" < /proc/1/environ | grep ^SPARKGLM_MODE"
  docker logs lloom-atlas-sparkglm-ennspark01 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "SERVING_MODE|serving mode|display carveout .* blocks|QUEUE_FIRST|OWNER_BATCH|LIVE_WIDTHS|thinking-off" | cut -c1-200 | head -6
  echo "=== serving $1 $(date -u +%T)"
}
cd $D/strict
boot strict || { echo "=== STRICT FAILED"; exit 1; }
python3 $Q/sq/greedy_eq.py run strict035 > greedy-strict035.log 2>&1; echo "greedy rc $?"
python3 $Q/sq/lp_dump.py strict035 > lpdump-strict035.log 2>&1; echo "lpdump rc $?"
python3 $Q/sq/greedy_eq.py compare strict033 strict035 > greedy-cmp.log 2>&1; python3 $Q/sq/lp_dump.py compare strict033 strict035 > lp-cmp.log 2>&1
tail -3 greedy-cmp.log | cut -c1-200; tail -6 lp-cmp.log | cut -c1-200
echo "=== STRICT DONE $(date -u +%T)"
