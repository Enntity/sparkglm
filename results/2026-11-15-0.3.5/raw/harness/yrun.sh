#!/bin/bash
# yrun.sh TAG [yexact args]: run the yield scenario on the engine now serving and keep its log.
T=$1; shift; cd ~/thinkoff; mkdir -p y
c=lloom-atlas-sparkglm-ennspark01
docker exec $c sh -c 'P=$(pgrep -f "spark serve" | head -1); tr "\0" "\n" < /proc/$P/environ | grep -E "PREFILL_YIELD|STRICT_OWNER|PAIR_LIVE" | sort | tr "\n" " "'; echo
M=$(date -u +%FT%T.%NZ)
python3 yexact.py y/$T.json "$@" 2>&1 | cut -c1-400
docker logs --since "$M" $c 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > y/$T.log
grep -c "Prefill chunk\|Queued prefill start\|Prefilled (single" y/$T.log | sed "s/^/prefill lines: /"
grep -E "ERROR|WARN|panicked|refus|diverged" y/$T.log | cut -c1-260 | sort | uniq -c | sort -rn | head -12
grep -E "Done:" y/$T.log | sed -E "s/.*Done: ([0-9]+ tokens \([a-z_]+\) [0-9.]+ tok\/s, TTFT=[0-9.]+ms).*/\1/" | head -12
