#!/bin/bash
# lone.sh TAG: three lone cold 16K-token prefills on the engine now serving; prints prompt tokens, TTFT and tok/s.
cd ~/thinkoff; T=$1
docker inspect lloom-atlas-sparkglm-ennspark01 --format "{{.Config.Image}}"
for i in 1 2 3; do python3 gbench.py --streams 1 --ctx 16000 --mode text --effort low --salt lone-$T-$i --out 32 2>&1 | tail -1 | python3 -c "
import sys,json
d=json.loads(sys.stdin.read()); p=d[\"prompt\"][0]; t=d[\"ttft_s\"][0]; print(\"prompt\", p, \"cached\", d[\"cached\"][0], \"ttft\", t, \"tok/s\", round(p/t))"; done
