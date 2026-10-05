#!/bin/bash
# sparkglm#30: carveout placement A/B on one binary (fix/carveout-placement), RigMark prefill 32K/64K x3 per arm.
[ -d $HOME/sparkglm-gpu.lock ] || { echo "take the pair first"; exit 1; }
P=~/sparkglm-pc/sparkglm-pin; export IMAGE=atlas-sparkglm:cvplace
measure() {
  cd ~/rigmark && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json --comparison-id cvp-$1-$(date +%s) \
    --runs 1 --prefill-runs 3 --prefill-depths 32768,65536 --skip-concurrency --extra-body "{\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}" \
    --label cvp-$1 --output ~/pf64/cvp-$1.json > ~/pf64/cvp-$1.log 2>&1
  echo "== $1"; grep -A3 "^prefill/" ~/pf64/cvp-$1.log | grep cold | sed "s/, warm.*//"
}
for arm in index index-only off; do
  case $arm in off) cv=0; export PROFILE=4x512k;; *) cv=1; export PROFILE=$HOME/pfprof/4x512k-$arm.json;; esac
  sed -i "s/^DISPLAY_CARVEOUT=.*/DISPLAY_CARVEOUT=$cv/" $P/.env
  cd $P && ./start.sh 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "image |== ready|rror" | tail -3
  docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -aoE "display carveout [0-9]+ MiB holds [0-9]+ buffers \([0-9]+ MiB\) → [0-9]+ blocks \(\+[0-9]+, \+[0-9.]+%\)|KV cache: [0-9]+ blocks" | head -2 | sed "s/^/   /"
  measure $arm
  cd $P && ./start.sh stop 2>&1 | tail -1
done
sed -i "s/^DISPLAY_CARVEOUT=.*/DISPLAY_CARVEOUT=1/" $P/.env
echo CVP-DONE
