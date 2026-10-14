#!/bin/bash
# go_bench.sh NEW_IMAGE: the SparkGLM release suite (go_final_bench.sh on 01, plus the 500+24 quality set, prose
# bursts and an arithmetic content probe) for NEW_IMAGE and the production image 5ec57d55, back to back on the
# 03/04 pair in production configuration. Outputs ~/final-bench/<label> and ~/field-final/<label>-<tokens> on 03.
NEW=${1:?usage: go_bench.sh NEW_IMAGE}
BASE=ghcr.io/enntity/atlas-sparkglm@sha256:5ec57d5522a95463b600a17471ce60712b315566e04c0a3bb6af1d84a1de0a4a
D=$HOME/sparkglm-dev; W=10.100.112.2; MR=$HOME/sparkglm-cleanroom/models; U=http://127.0.0.1:8893; M=glm-5.3-flash-atlas
[ -d $HOME/sparkglm-gpu.lock ] || { echo "take the 03/04 marker lock first"; exit 1; }
memwait() { for i in $(seq 60); do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); b=$(ssh -n $W "awk \"/MemAvailable/{print int(\\\$2/1048576)}\" /proc/meminfo")
    [ "$a" -ge 100 ] && [ "$b" -ge 100 ] && return 0; sleep 2; done; echo "memwait: only $a / $b GiB available"; }
run() {  # run TAG IMAGE
  docker rm -f atlas-sparkglm-rank0 >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank1 >/dev/null 2>&1"; memwait
  ssh $W "bash -s -- --rank 1 --leader-address 10.100.112.1 --model-root $MR --image $2 --profile $D/prod_4x1m.json" < $D/start-node-img.sh >/dev/null
  bash $D/start-node-img.sh --rank 0 --leader-address 10.100.112.1 --model-root $MR --image $2 --profile $D/prod_4x1m.json >/dev/null
  for i in $(seq 150); do curl -sf $U/health >/dev/null && break
    if [ "$(docker inspect -f {{.State.Status}} atlas-sparkglm-rank0 2>/dev/null)" != running ]; then echo "== $1 FAILED TO START"; exit 1; fi
    sleep 5; done
  echo "== $1 up ($2): $(docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "agree on [0-9]+|carveout not lent" | tr "\n" " ")"
}
arith() {  # the soak's short questions, unstreamed, content and reasoning both shown
python3 - "$1" <<"PY"
import json, random, sys, urllib.request
rng = random.Random(9); out = []
for _ in range(40):
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    body = {"model": "glm-5.3-flash-atlas", "messages": [{"role": "user", "content": f"What is {a} times {b}? Number only."}],
            "max_tokens": 12, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8893/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    m = r["choices"][0]["message"]
    out.append({"q": [a, b], "content": m.get("content"), "reasoning": m.get("reasoning_content"), "finish": r["choices"][0]["finish_reason"]})
ok = sum(str(o["q"][0] * o["q"][1]) in (o["content"] or "").replace(",", "") for o in out)
json.dump(out, open(f"arith-{sys.argv[1]}.json", "w"), indent=1)
print(f"arith {sys.argv[1]}: {ok}/{len(out)} exact; empty content {sum(not o['content'] for o in out)}")
PY
}
bench() {  # bench LABEL IMAGE
  L=$1; O=$HOME/final-bench/$L; mkdir -p $O; T=$(date +%s); run $L $2
  cd $D; arith $L; python3 replay_fill.py $L
  cd ~/rigmark
  ./rigmark run --base-url $U --model auto --metadata metadata.json --comparison-id $L --runs 3 --prefill-runs 2 \
    --prefill-depths 8192,32768,65536 --skip-concurrency --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
    --label $L-decode --output $O/rig-decode.json > $O/rig-decode.log 2>&1; echo "rig-decode exit $?"
  ./rigmark run --base-url $U --model auto --metadata metadata.json --comparison-id $L --runs 1 --skip-prefill \
    --concurrency 1,2,3,4 --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
    --staggered 2,4 --staggered-runs 2 --staggered-depth 32768 --staggered-incumbent-tokens 1024 --staggered-arrival-tokens 256 \
    --staggered-delay 1.0 --staggered-workload prose --label $L-conc --output $O/rig-conc.json > $O/rig-conc.log 2>&1; echo "rig-conc exit $?"
  bash $D/go_field.sh $L $U $M $L 16384 5
  bash $D/go_field.sh $L $U $M $L 32768 3
  cd ~/bench-tf && MATRIX_BASE_URL=$U MATRIX_MODEL=$M MATRIX_SALT=mb-$L-$T MATRIX_IGNORE_EOS=1 python3 run_matrix.py $O/matrix matrix > $O/matrix.log 2>&1; echo "matrix exit $?"
  mkdir -p $O/quality && cd $O/quality && timeout 1800 python3 $D/quality_probe_glm.py $L 4 2>&1 | tail -2
  cd $O && timeout 3600 python3 $D/qprobe_big.py $L 500 24 4 2>&1 | tail -2
  cd $D && python3 step_bursts.py $L $U $M 2>&1 | tail -1; mv -f ~/step0/bursts-$L.json $O/ 2>/dev/null
  docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > $O/rank0.log; ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1" | sed "s/\x1b\[[0-9;]*m//g" > $O/rank1.log
  echo "BENCH-DONE $L"
}
bench vpc-03f0ecc $NEW
bench prod-5ec57d5 $BASE
docker rm -f atlas-sparkglm-rank0 >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank1 >/dev/null 2>&1"
python3 - <<"PY"
import json, os
H = os.path.expanduser("~")
load = lambda p: [json.loads(l)["out"] for l in open(p)]
new = load(f"{H}/final-bench/vpc-03f0ecc/qbig-vpc-03f0ecc.jsonl")
for name, p in (("prod on 03/04", f"{H}/final-bench/prod-5ec57d5/qbig-prod-5ec57d5.jsonl"), ("RC4 on 01/02", f"{H}/sparkglm-dev/qbig-rc4-on.jsonl")):
    ref = load(p); same = sum(a == b for a, b in zip(new, ref))
    print(f"qbig outputs identical to {name}: {same}/{len(ref)}")
PY
echo GO-BENCH-DONE
