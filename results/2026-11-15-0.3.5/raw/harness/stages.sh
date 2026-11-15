#!/bin/bash
# stages.sh TAG: today's qualification history in order, with three lone cold 19K prefills after each stage.
cd ~/thinkoff; T=$1; c=lloom-atlas-sparkglm-ennspark01; mkdir -p st-$T
curls() { for n in 6 4; do curl -s -m 60 -H "Authorization: Bearer sk-local-llm" -H "content-type: application/json" localhost:8893/v1/chat/completions -d "{\"model\":\"glm-5.3-flash-atlas\",\"temperature\":0,\"max_tokens\":$n,\"logprobs\":true,\"top_logprobs\":1,\"reasoning_effort\":\"low\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}]}" > /dev/null; done; }
chk() { echo "--- after $1 $(date -u +%T)"; M=$(date -u +%FT%TZ); ./lone.sh $T-$1 | tail -3
  docker logs --since $M $c 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > st-$T/lone-$1.log
  grep -ciE "evict|spill|swap|tier" st-$T/lone-$1.log | sed "s/^/evict-ish lines: /"; }
docker inspect $c --format "{{.Config.Image}}"; chk fresh
./yrun.sh st-$T-s > /dev/null 2>&1; curls; ./yrun.sh st-$T-lp --lp > /dev/null 2>&1; ./yrun.sh st-$T-cold --lp --salt 1 > /dev/null 2>&1; chk yexact
python3 verify.py http://127.0.0.1:8893 > /dev/null 2>&1; chk verify
./cells.sh st-$T > /dev/null 2>&1; chk cells
python3 jexact.py --n 12 --out 300 --salt st$T > /dev/null 2>&1; chk jexact
python3 soak.py --secs 180 --streams 4 --seed 7 > /dev/null 2>&1; chk soak
echo "=== STAGES END $(date -u +%T)"
