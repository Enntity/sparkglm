#!/usr/bin/env bash
# RiNGSiDE TP2 cell set against Atlas (code-full, then prose concurrency + staggered arrivals).
T=${1:-atlas}
cd ~/rigmark
./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json \
  --comparison-id atlas-vs-ringside-tp2 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6 \
  --concurrency-runs 2 --extra-body "{\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}" \
  --label $T-code-full --prefill-depths 8192,32768,65536 --output ~/rig-$T-code-full.json > ~/rig-$T-code-full.log 2>&1
./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json \
  --comparison-id atlas-vs-ringside-tp2 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6 \
  --concurrency-runs 2 --extra-body "{\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}" \
  --label $T-prose-concurrency --concurrency-workload prose --skip-prefill \
  --staggered 2,4,6 --staggered-runs 2 --staggered-depth 32768 --staggered-incumbent-tokens 1024 \
  --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose --output ~/rig-$T-prose.json > ~/rig-$T-prose.log 2>&1
echo DONE > ~/rig-$T.done
