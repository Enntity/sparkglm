#!/bin/bash
# final035y.sh: release checks of the serving SparkGLM 0.3.5 image with arrival-order prefill and the yield off (run on a freshly started engine).
cd ~/thinkoff; mkdir -p f035g; T=sg035g-1115; c=lloom-atlas-sparkglm-ennspark01
docker inspect $c --format "{{.Config.Image}}"
curls() { for n in 6 4; do curl -s -m 60 -H "Authorization: Bearer sk-local-llm" -H "content-type: application/json" localhost:8893/v1/chat/completions -d "{\"model\":\"glm-5.3-flash-atlas\",\"temperature\":0,\"max_tokens\":$n,\"logprobs\":true,\"top_logprobs\":1,\"reasoning_effort\":\"low\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}]}" > /dev/null; done; }
echo "=== verify $(date -u +%T)"; python3 verify.py http://127.0.0.1:8893 > f035g/verify.out 2>&1; tail -1 f035g/verify.out; grep -c "^ok" f035g/verify.out
echo "=== cells $(date -u +%T)"; ./cells.sh sg035g > /dev/null 2>&1; cut -c1-150 cells/sg035g.out
echo "=== jexact $(date -u +%T)"; python3 jexact.py --n 12 --out 300 --salt f035g > f035g/jexact.out 2>&1; cut -c1-300 f035g/jexact.out
echo "=== soak $(date -u +%T)"; M=$(date -u +%FT%TZ); python3 soak.py --secs 180 --streams 4 --seed 7 > f035g/soak.out 2>&1; cut -c1-330 f035g/soak.out
docker logs --since $M $c 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -cE " ERROR |panick"
echo "=== qual $(date -u +%T)"; ./qual2.sh $T 2>&1 | cut -c1-200
echo "=== rig $(date -u +%T)"; ~/grind/rig.sh $T > f035g/rig.out 2>&1; grep -E "^/decode/(prose|code|structured)/decode_tokens_per_second/median|^/prefill/16384/cold/effective_prefill_tokens_per_second/median" f035g/rig.out
echo "errors all: $(docker logs $c 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -cE " ERROR |panick")"
echo "=== FINAL END $(date -u +%T)"
