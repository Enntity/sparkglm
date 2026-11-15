#!/bin/bash
# stages2.sh TAG: the qualification and RigMark stages, with lone cold prefills after each.
cd ~/thinkoff; T=$1; c=lloom-atlas-sparkglm-ennspark01; mkdir -p st-$T
chk() { echo "--- after $1 $(date -u +%T)"; M=$(date -u +%FT%TZ); ./lone.sh $T-$1 | tail -3
  docker logs --since $M $c 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > st-$T/lone-$1.log; }
docker inspect $c --format "{{.Config.Image}}"; chk before
./qual2.sh sg-st-$T > st-$T/qual.out 2>&1; tail -3 st-$T/qual.out | cut -c1-120; chk qual
~/grind/rig.sh sg-st-$T > st-$T/rig.out 2>&1; grep -E "^/prefill/16384/cold/effective_prefill_tokens_per_second/median" st-$T/rig.out; chk rig
echo "=== STAGES2 END $(date -u +%T)"
