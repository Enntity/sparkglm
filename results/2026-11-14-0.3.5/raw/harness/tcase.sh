#!/bin/bash
# tcase.sh TAG gbench-args...: run one gbench case on the instrumented engine and summarize its step trace.
T=$1; shift; cd ~/thinkoff; mkdir -p tc
python3 gbench.py "$@" --out 64 > /dev/null   # warm contexts and CUDA graphs for this shape
M=$(date -u +%FT%T.%NZ); sleep 0.5
python3 gbench.py "$@" | cut -c1-230
docker logs --since "$M" lloom-atlas-sparkglm-ennspark01 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > tc/$T.log
python3 steps2.py tc/$T.log
