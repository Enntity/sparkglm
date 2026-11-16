#!/bin/bash
# vision.sh: after the Strict check, back to Spark mode (as the recipe installs it) and image/video output of 0.3.5 against the stored 0.3.3 run.
set -u; export PATH=$HOME/.local/bin:$PATH; D=~/sg-035
IMG=ghcr.io/enntity/atlas-sparkglm@sha256:ab56ad3e0879da5b039903fede4838d762f70d991509f2ad3598baca2446b872
until grep -q -E "STRICT DONE|STRICT FAILED" $D/strict.out; do sleep 15; done
lloom suspend glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error" | head -1
python3 $D/mode-conf.py $IMG spark && ssh RANK1_FABRIC_ADDR "python3 /tmp/mode-conf.py $IMG spark"
sleep 45
for try in 1 2 3 4 5; do out=$(lloom resume glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error" | head -2); echo "$out"; echo "$out" | grep -q Error || break; sleep 45; done
for i in $(seq 90); do curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas && break; sleep 5; done
docker exec lloom-atlas-sparkglm-ennspark01 sh -c "tr \"\\0\" \"\\n\" < /proc/1/environ | grep ^SPARKGLM_MODE"
cd ~/kv-reclaim && python3 vision_eq.py run sg035 > veq-sg035.log 2>&1; echo "run rc $?"; python3 vision_eq.py compare sg033 sg035
docker logs lloom-atlas-sparkglm-ennspark01 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "holds [0-9] buffers \([0-9]+ MiB\) → [0-9]+ blocks" | tail -1
echo "=== VISION DONE $(date -u +%T)"
