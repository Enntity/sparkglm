#!/bin/bash
# glm_suite.sh ARM : GLM checks against the server on 127.0.0.1:8893 (outputs in ~/vis-ab).
A=$1; cd ~/vis-ab
(while true; do echo "$(date +%T) $(awk "/MemAvailable/{print \$2}" /proc/meminfo)"; sleep 5; done > mem-glm-$A.log) & S=$!
echo "-- prompt-logprob hash"; timeout 600 python3 ~/sparkglm-dev/lp_repeat.py glm-$A 2100 2 2>&1 | tail -1; python3 -c "import json; print(\"hash\", [r[\"hash\"] for r in json.load(open(\"lp-glm-$A.json\"))[\"runs\"]])"
echo "-- prefix-cache test"; timeout 600 python3 ~/sparkglm-dev/pc_test.py glm-$A 2>&1 | tail -2
echo "-- repeat repro"; timeout 300 python3 ~/sparkglm-dev/repeat_repro.py glm-$A 2>&1 | tail -2
echo "-- long tool call"; timeout 900 python3 - <<"PY"
import json, urllib.request
tools = [{"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]
body = {"model": "glm-5.3-flash-atlas", "temperature": 0, "max_tokens": 6000, "tools": tools, "messages": [{"role": "user", "content": "Use write_file to create big.py containing 150 small, distinct, fully written Python functions named f1..f150, each with a docstring. Write the whole file in one call."}]}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8893/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900))
c = r["choices"][0]; tc = c["message"].get("tool_calls") or []; args = json.loads(tc[0]["function"]["arguments"]) if tc else {}
print("long tool call: finish", c["finish_reason"], "tokens", r["usage"]["completion_tokens"], "tool_calls", len(tc), "has f150", "def f150" in args.get("content", ""))
PY
echo "-- speed probe"; timeout 1200 python3 ~/sparkglm-dev/multi_probe.py glm_$A 2>&1 | tail -1
echo "-- vision gate"; cd vpc-glm; for ph in race2 conv swap; do timeout 900 python3 vpc_gate.py $A $ph 2>&1 | tail -6; done; cd ..
kill $S; echo "-- min MemAvailable rank0 kB: $(sort -k2 -n mem-glm-$A.log | head -1)"
for c in 0 1; do if [ $c = 0 ]; then docker logs atlas-sparkglm-rank0 2>&1; else ssh -n RANK1_HOST "docker logs atlas-sparkglm-rank1 2>&1"; fi | sed "s/\x1b\[[0-9;]*m//g" > glm-$A.rank$c.log; done
echo "-- log scan: rank0 panics $(grep -ciE "panicked" glm-$A.rank0.log) errors $(grep -cE " ERROR " glm-$A.rank0.log); rank1 panics $(grep -ciE "panicked" glm-$A.rank1.log) errors $(grep -cE " ERROR " glm-$A.rank1.log); memguard: $(tail -1 ~/memguard.log)"
