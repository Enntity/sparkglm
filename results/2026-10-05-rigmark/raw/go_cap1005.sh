#!/bin/bash
# Production f047a0c5e796 (carveout latent-only): field guide (discarded warmup + 5) and RigMark staggered arrivals at 2/4 (prefill-first + decode-first), for the graphic.
O=~/caps-prod1005; mkdir -p $O/field; cd ~/bench-tf; T=$(date +%s)
curl -sf -m 5 127.0.0.1:8893/health >/dev/null || { echo "production not healthy"; exit 1; }
for n in discarded-warmup field-guide-r1 field-guide-r2 field-guide-r3 field-guide-r4 field-guide-r5; do
  python3 four_stream_video.py capture --base-url http://127.0.0.1:8893 --model glm-5.3-flash-atlas --output $O/field/$n.json \
    --streams 4 --max-tokens 400 --prompt-tokens 16384 --stagger-ms 1000 --prompt-style field-guide \
    --prompt-salt fg-p1005-$T-$n --cache-salt fg-p1005-$T-$n --recipe-label "SparkGLM Atlas 2026-10-05" --engine-commit 72588541 \
    --timeout-s 900 > $O/field/$n.log 2>&1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1].split(\"/\")[-1], d[\"wall_s\"], d[\"total_completion_tokens\"])" $O/field/$n.json
done
cd ~/rigmark && ./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json --comparison-id atlas-stag-1005 \
  --runs 1 --skip-prefill --skip-concurrency --extra-body "{\"chat_template_kwargs\":{\"reasoning_effort\":\"low\"}}" \
  --staggered 2,4 --staggered-runs 2 --staggered-depth 32768 --staggered-incumbent-tokens 1024 --staggered-arrival-tokens 256 \
  --staggered-delay 1.0 --staggered-workload prose --label atlas-1005-stag --output ~/rig-atlas-1005-stag.json > ~/rig-atlas-1005-stag.log 2>&1
echo "rigmark exit $?"; echo CAP-DONE
