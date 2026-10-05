#!/bin/bash
# 64K cold-prefill regression (Enntity/sparkglm#30): RigMark prefill 32K/64K x3 per arm on the idle pair.
#   cur1: current release (35824765d542) carveout on, 0.91   cur0: same, carveout off   old: 2026-09-30 release (43eed6283b96, sparkglm 5db6d0c) at 0.93
[ -d $HOME/sparkglm-gpu.lock ] || { echo "take the pair first"; exit 1; }
P=~/sparkglm-pc/sparkglm-pin; O=~/sparkglm-pc/pf64-old
[ -d $O ] || git -C $P worktree add -q --detach $O 5db6d0c
grep -v -E "^(DISPLAY_CARVEOUT|GPU_MEMORY_UTILIZATION)=" $P/.env > $O/.env; echo GPU_MEMORY_UTILIZATION=0.93 >> $O/.env
measure() {
  cd ~/rigmark && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json --comparison-id pf64-$1-$(date +%s) \
    --runs 1 --prefill-runs 3 --prefill-depths 32768,65536 --skip-concurrency --extra-body "{\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}" \
    --label pf64-$1 --output ~/pf64/$1.json > ~/pf64/$1.log 2>&1
  echo "== $1"; grep -A3 "^prefill/" ~/pf64/$1.log | grep cold | sed "s/, warm.*//"
}
mkdir -p ~/pf64
for arm in cur1 cur0 old; do
  case $arm in cur1) D=$P; sed -i "s/^DISPLAY_CARVEOUT=.*/DISPLAY_CARVEOUT=1/" $P/.env;; cur0) D=$P; sed -i "s/^DISPLAY_CARVEOUT=.*/DISPLAY_CARVEOUT=0/" $P/.env;; old) D=$O;; esac
  cd $D && ./start.sh 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "image |== ready|rror" | tail -3
  echo "   $(docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -aoE "KV cache: [0-9]+ blocks" | head -1)"
  measure $arm
  cd $D && ./start.sh stop 2>&1 | tail -1
done
sed -i "s/^DISPLAY_CARVEOUT=.*/DISPLAY_CARVEOUT=1/" $P/.env
echo PF64-DONE
