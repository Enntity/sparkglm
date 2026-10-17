#!/bin/bash
# gates2.sh TAG BIN ENVFILE: pre-deploy gates on the production 4x1m profile + ENVFILE (pair held; run inside serve_lock.sh).
set -u; T=$1; BIN=$2; EF=$3; B=~/rel-sparkqwen/bench; G=~/glm-fp4l; D=$G/g-$T; mkdir -p $D; cd $D; W=RANK1_HOST
export SQ_MODEL=glm-5.3-flash-atlas
python3 - "$G/profiles/4x1m.json" "$D/4x1m.json" "$EF" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for l in open(sys.argv[3]):
    l = l.strip()
    if l and not l.startswith("#"):
        k, v = l.split("=", 1); p["environment"][k] = v
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
ssh -n $W "mkdir -p $D"; rsync -a $D/4x1m.json $W:$D/
echo "##### $T pre-deploy gates (4x1m + $EF) $(date +%T)"; grep -v "^#" $EF
bash $G/up.sh $T fp4l $BIN $D/4x1m.json 0.91 || { bash ~/fp4l/scan.sh $D N; exit 1; }
(while true; do echo "$(date +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 5; done > mem0.log) & S=$!
echo "-- quality"; timeout 1800 python3 $B/quality_probe.py $T 4 2>&1 | tail -3
echo "-- tools"; timeout 900 python3 $B/tool_probe.py $T 3 2>&1 | tail -4
for c in 1 4; do echo "-- greedy C$c"; timeout 900 python3 $B/greedy_eq.py run $T-c$c $c 2>&1 | tail -1; done
echo "-- greedy C4 prefix"; timeout 900 python3 $B/greedy_eq.py run $T-c4p 4 --prefix 2>&1 | tail -1
echo "-- lp hash"; timeout 900 python3 ~/sparkglm-dev/lp_repeat.py glm-$T 2100 2 2>&1 | tail -1; python3 -c "import json; print('hash', [r['hash'] for r in json.load(open('lp-glm-$T.json'))['runs']])"
echo "-- prefix-cache test"; timeout 600 python3 ~/sparkglm-dev/pc_test.py glm-$T 2>&1 | tail -1
echo "-- four ~204K sessions fill"; timeout 1500 python3 ~/sparkglm-dev/pc_fill.py glm-$T 2>&1 | tail -2
kill $S; echo "-- min MemAvailable rank0 kB: $(sort -k2 -n mem0.log | head -1)"
bash ~/fp4l/scan.sh $D N
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
echo "##### $T gates done $(date -Is)"
