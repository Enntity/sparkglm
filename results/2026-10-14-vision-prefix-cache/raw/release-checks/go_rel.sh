#!/bin/bash
# go_rel.sh NEW_IMAGE: release checks for the vision prefix-cache image on the 03/04 pair, production
# configuration (start-node-img.sh: util 0.91, display carveout, NVMe prefix + SSM tiers, KV shard, 1M
# context; profile = the shipped 4x1m). Runs from ennspark03 with the 03/04 marker lock held.
#   base   production image 5ec57d55: startup footprint, prompt-logprob hash, decode speed
#   new    NEW_IMAGE: the same, then the long tool call, the cache test, the vision checks, a 20 min soak
#   fill   NEW_IMAGE fresh: an image conversation, five fresh-seed 4x204K fill rounds (the fill check and
#          the eviction pressure), then the image turn again: the NVMe tier must restore it
# MemAvailable of both hosts every second -> mem-rel-rank{0,1}.log; arm times -> arms-rel.txt.
NEW=${1:?usage: go_rel.sh NEW_IMAGE}
BASE=ghcr.io/enntity/atlas-sparkglm@sha256:5ec57d5522a95463b600a17471ce60712b315566e04c0a3bb6af1d84a1de0a4a
D=$HOME/sparkglm-dev; W=10.100.112.2; MR=$HOME/sparkglm-cleanroom/models
[ -d $HOME/sparkglm-gpu.lock ] || { echo "take the 03/04 marker lock first"; exit 1; }
cd $D
memwait() { for i in $(seq 60); do a=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); b=$(ssh -n $W "awk \"/MemAvailable/{print int(\\\$2/1048576)}\" /proc/meminfo")
    [ "$a" -ge 100 ] && [ "$b" -ge 100 ] && return 0; sleep 2; done; echo "memwait: only $a / $b GiB available"; }
stamp() { echo "$1 $(date -Is)" >> $D/arms-rel.txt; }
run() {  # run TAG IMAGE
  [ -n "$CUR" ] && stamp "$CUR-end"
  docker rm -f atlas-sparkglm-rank0 >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank1 >/dev/null 2>&1"; memwait
  ssh $W "bash -s -- --rank 1 --leader-address 10.100.112.1 --model-root $MR --image $2 --profile $D/prod_4x1m.json" < $D/start-node-img.sh >/dev/null
  bash $D/start-node-img.sh --rank 0 --leader-address 10.100.112.1 --model-root $MR --image $2 --profile $D/prod_4x1m.json >/dev/null
  for i in $(seq 150); do curl -sf http://127.0.0.1:8893/health >/dev/null && break
    if [ "$(docker inspect -f {{.State.Status}} atlas-sparkglm-rank0 2>/dev/null)" != running ] || [ "$(ssh -n $W docker inspect -f {{.State.Status}} atlas-sparkglm-rank1 2>/dev/null)" != running ]; then
      echo "== $1 FAILED TO START"; docker logs --tail 8 atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-240; ssh -n $W "docker logs --tail 8 atlas-sparkglm-rank1 2>&1" | sed "s/\x1b\[[0-9;]*m//g" | cut -c1-240; exit 1; fi
    sleep 5; done
  CUR=$1; stamp "$1-start"
  echo "== $1 up ($2): $(docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -oE "agree on [0-9]+" | head -1)"
}
logs() { docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" > $D/rel-$1.rank0.log; ssh -n $W "docker logs atlas-sparkglm-rank1 2>&1" | sed "s/\x1b\[[0-9;]*m//g" > $D/rel-$1.rank1.log
  echo "-- $1 footprint:"; for r in 0 1; do grep -ahE "pre-KV \+|fits [0-9]+ blocks|vision weights not loaded" $D/rel-$1.rank$r.log | sed "s/^.*INFO //" | cut -c1-200; done
  echo "-- $1 log errors: rank0 $(grep -ciE "panicked|ERROR " $D/rel-$1.rank0.log) rank1 $(grep -ciE "panicked|ERROR " $D/rel-$1.rank1.log)"; }
longtool() { python3 - <<"PY"
import json, urllib.request
tools = [{"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]
body = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 6000, "tools": tools,
        "messages": [{"role": "user", "content": "Use write_file to create big.py containing 150 small, distinct, fully written Python functions named f1..f150, each with a docstring. Write the whole file in one call."}]}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8893/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900))
c = r["choices"][0]; tc = c["message"].get("tool_calls") or []
args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
print("long tool call: finish", c["finish_reason"], "completion_tokens", r["usage"]["completion_tokens"], "tool_calls", len(tc), "has f150", "def f150" in args.get("content", ""))
PY
}
pkill -f "[v]pc-mem1s.sh"; ssh -n $W 'pkill -f "[v]pc-mem1s.sh"'
setsid nohup bash $D/vpc-mem1s.sh $D/mem-rel-rank0.log >/dev/null 2>&1 < /dev/null &
ssh -n $W "setsid nohup bash \$HOME/sparkglm-dev/vpc-mem1s.sh /tmp/mem-rel-rank1.log >/dev/null 2>&1 < /dev/null &"

run base $BASE
timeout 600 python3 lp_repeat.py rel-base 2100 1 | tail -1
python3 multi_probe.py rel-base | tail -1
logs base

run new $NEW
timeout 600 python3 lp_repeat.py rel-new 2100 1 | tail -1
python3 multi_probe.py rel-new | tail -1
longtool
python3 pc_test.py rel-new 2>&1 | tail -3
python3 vpc_gate.py rel race2; python3 vpc_gate.py rel conv; python3 vpc_gate.py rel swap
python3 soak.py rel 20
logs new

run fill $NEW
python3 vpc_gate.py relnvme conv
for s in 1 2 3 4 5; do FILL_SEED=$((9000 + s)) python3 pc_fill_fresh.py rel-fill$s 2>&1 | tail -1; done
python3 - <<"PY"
import json, vpc_gate
saved = json.load(open("conv-relnvme.json")); prev = saved["turns"][4]
r = vpc_gate.stream(saved["messages"][:prev["messages"]], max_tokens=80); vpc_gate.log("image turn 5 after fills", r)
print("same text as before the fills:", r["text"] == prev["text"])
json.dump({"after": r, "before": prev}, open("nvme-rel.json", "w"), indent=1)
PY
logs fill
grep -ahE "NVMe prefix restore|NVMe spill" $D/rel-fill.rank0.log | sed "s/^.*INFO //" | tail -3 | cut -c1-260
stamp "$CUR-end"
docker rm -f atlas-sparkglm-rank0 >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank1 >/dev/null 2>&1"
pkill -f "[v]pc-mem1s.sh"; ssh -n $W 'pkill -f "[v]pc-mem1s.sh"'; scp -q $W:/tmp/mem-rel-rank1.log $D/
python3 - <<"PY"
arms = {}
for l in open("arms-rel.txt"):
    k, t = l.split(); n, e = k.rsplit("-", 1); arms.setdefault(n, {})[e] = t[:19]
for n, w in arms.items():
    row = []
    for f in ("mem-rel-rank0.log", "mem-rel-rank1.log"):
        v = [int(x.split()[1]) for x in open(f) if len(x.split()) == 2 and w["start"] <= x.split()[0] <= w.get("end", "9")]
        row.append(round(min(v) / 1048576, 2) if v else None)
    print(f"min MemAvailable GiB {n}: rank0 {row[0]} rank1 {row[1]}")
PY
echo REL-DONE
