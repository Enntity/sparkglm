#!/bin/bash
# pfab.sh TAG BIN [ENVFILE] [SIZES]: prefill-only arm on 4x512k (run inside serve_lock.sh). PF_ARGS="--flag=v ..." overrides server argv.
set -u; T=$1; BIN=$2; EF=${3:-}; SZ=${4:-32k,128k}; G=~/glm-fp4l; D=$G/p-$T; mkdir -p $D; cd $D; W=RANK1_HOST
python3 - "$G/profiles/4x512k.json" "$D/4x512k.json" "$EF" "${PF_ARGS:-}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); ef = sys.argv[3]
if ef:
    for l in open(ef):
        l = l.strip()
        if l and not l.startswith("#"):
            k, v = l.split("=", 1); p["environment"][k] = v
for a in sys.argv[4].split():
    k = a.split("=", 1)[0]
    p["server_argv"] = [x for x in p["server_argv"] if x.split("=", 1)[0] != k] + [a]
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
ssh -n $W "mkdir -p $D"; rsync -a $D/4x512k.json $W:$D/
echo "##### $T prefill arm (env ${EF:-none}, args ${PF_ARGS:-none}) $(date +%T)"; [ -n "$EF" ] && grep -v "^#" $EF
bash $G/up.sh $T fp4l $BIN $D/4x512k.json 0.91 || { bash ~/fp4l/scan.sh $D P; exit 1; }
timeout 1800 python3 $G/kgate/prefill.py --base http://127.0.0.1:8893/v1 --model glm-5.3-flash-atlas --sizes $SZ --skip-thinking --out-tokens 32 --seed 11 2>&1 | grep -A3 prompt_tok | tee $D/prefill.txt
if [ "${LP:-0}" = 1 ]; then timeout 900 python3 ~/sparkglm-dev/lp_repeat.py glm-$T 2100 2 2>&1 | tail -1; python3 -c "import json; print('lp hash', [r['hash'] for r in json.load(open('lp-glm-$T.json'))['runs']])"; fi
bash ~/fp4l/scan.sh $D P
for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done
echo "##### $T prefill done $(date +%T)"
