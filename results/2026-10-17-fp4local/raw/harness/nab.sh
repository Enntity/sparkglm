#!/bin/bash
# nab.sh TAG BIN ENVFILE: faithful-vs-production arm on 4x512k (run inside serve_lock.sh): cold prefill 32K, kindling decode (6 runs, series nab), greedy C1 vs C4.
set -u; T=$1; BIN=$2; EF=$3; G=~/glm-fp4l; D=$G/n-$T; mkdir -p $D; cd $D; W=RANK1_HOST; B=~/rel-sparkqwen/bench
python3 - "$G/profiles/4x512k.json" "$D/4x512k.json" "$EF" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for l in open(sys.argv[3]):
    l = l.strip()
    if l and not l.startswith("#"):
        k, v = l.split("=", 1); p["environment"][k] = v
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
ssh -n $W "mkdir -p $D"; rsync -a $D/4x512k.json $W:$D/
echo "##### $T native arm $(date +%T)"; grep -v "^#" $EF | tr "\n" " "; echo
bash $G/up.sh $T fp4l $BIN $D/4x512k.json 0.91 || { bash ~/fp4l/scan.sh $D N; exit 1; }
timeout 900 python3 $G/kgate/prefill.py --base http://127.0.0.1:8893/v1 --model glm-5.3-flash-atlas --sizes 32k --skip-thinking --out-tokens 32 --seed 7 2>&1 | grep -A2 prompt_tok
GATE_URL=http://127.0.0.1:8893 GATE_MODEL=glm-5.3-flash-atlas timeout 2400 python3 $G/kgate/bench/bench.py run --label $T --series nab --workloads code,prose,structured --runs 6 --warmup 1 --out $D/kb-$T.json 2>&1 | grep -E "^ *(code|prose|structured)"
export SQ_MODEL=glm-5.3-flash-atlas
for c in 1 4; do timeout 900 python3 $B/greedy_eq.py run $T-c$c $c >/dev/null 2>&1; done
echo "-- greedy C4 vs C1: $(python3 $B/greedy_eq.py compare $T-c4 $T-c1 2>&1 | tail -1)"
bash ~/fp4l/scan.sh $D N
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
echo "##### $T done $(date +%T)"
