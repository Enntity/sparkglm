#!/usr/bin/env python3
"""Harder prefix-cache correctness checks against a running engine.

  1. four concurrent multi-turn sessions (distinct documents), exact answers
  2. a tool-call turn, then a question that needs the document and the tool result
  3. a partial hit: an earlier message edited, needles before and after the edit
  4. long greedy output: cold vs cache hit on the identical prompt
  5. thinking on: cached-token count across turns
Usage: pc_hard.py TAG
"""
import json, random, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8893/v1/chat/completions"
MODEL = "glm-5.3-flash-atlas"
NO_THINK = {"enable_thinking": False, "thinking": False}
out = {"tag": sys.argv[1]}


def call(messages, max_tokens=64, think=False, tools=None, tool_choice=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    if not think:
        body["chat_template_kwargs"] = NO_THINK
    else:
        body["chat_template_kwargs"] = {"reasoning_effort": "low"}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = None; calls = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                delta = c.get("delta") or {}
                if (delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t0
                text += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["id"] = tc.get("id") or slot["id"]
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
    return {"ttft": round(ttft or -1, 2), "text": text.strip(), "calls": list(calls.values()),
            "prompt": (usage or {}).get("prompt_tokens"),
            "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}


def document(n, seed):
    rng = random.Random(seed)
    rows, facts = [], {}
    for i in range(n):
        name = f"unit-{seed}-{i:05d}"
        code = rng.randint(100000, 999999)
        rows.append(f"Record {i}: {name} has access code {code} and owner team-{rng.randint(1, 99)}.")
        facts[name] = code
    return rows, facts


def ask(key):
    return {"role": "user", "content": f"What is the access code of {key}? Answer with the number only."}


# 1. Four concurrent sessions.
def session(idx, results):
    rows, facts = document(1200, 100 + idx)
    keys = random.Random(idx).sample(sorted(facts), 4)
    convo = [{"role": "system", "content": "Answer questions about these records exactly.\n\n" + "\n".join(rows)}]
    for turn, key in enumerate(keys, 1):
        convo.append(ask(key))
        r = call(convo)
        r.update(session=idx, turn=turn, correct=str(facts[key]) in r["text"])
        results.append(r)
        convo.append({"role": "assistant", "content": r["text"]})

res = []
threads = [threading.Thread(target=session, args=(i, res)) for i in range(4)]
[t.start() for t in threads]; [t.join() for t in threads]
res.sort(key=lambda r: (r["session"], r["turn"]))
out["concurrent"] = res
for r in res:
    print("concurrent s%d t%d" % (r["session"], r["turn"]), r["ttft"], r["prompt"], r["cached"], r["correct"], flush=True)

# 2. Tool call turn.
rows, facts = document(1000, 21)
keys = sorted(facts)
tools = [{"type": "function", "function": {"name": "lookup_team_lead", "description": "Return the lead of a team.",
          "parameters": {"type": "object", "properties": {"team": {"type": "string"}}, "required": ["team"]}}}]
convo = [{"role": "system", "content": "You can call tools. Records:\n\n" + "\n".join(rows)},
         {"role": "user", "content": "Use the tool to look up the lead of team-7."}]
r1 = call(convo, max_tokens=200, tools=tools, tool_choice="required")
tc = r1["calls"][0] if r1["calls"] else {"id": "call_x", "name": "lookup_team_lead", "arguments": '{"team":"team-7"}'}
convo.append({"role": "assistant", "content": "", "tool_calls": [{"id": tc["id"] or "call_x", "type": "function",
              "function": {"name": tc["name"], "arguments": tc["arguments"]}}]})
convo.append({"role": "tool", "tool_call_id": tc["id"] or "call_x", "content": "The lead of team-7 is Ada Okonkwo."})
key = keys[123]
convo.append({"role": "user", "content": f"Who leads team-7, and what is the access code of {key}? "
              "Reply as: NAME; CODE"})
r2 = call(convo, max_tokens=64, tools=tools)
tool_ok = bool(r1["calls"]) and "team-7" in r1["calls"][0]["arguments"] and "Ada Okonkwo" in r2["text"] and str(facts[key]) in r2["text"]
out["tool"] = {"first": r1, "second": r2, "correct": tool_ok}
print("tool", r1["calls"][:1], "|", r2["ttft"], r2["prompt"], r2["cached"], repr(r2["text"][:60]), tool_ok, flush=True)

# 3. Partial hit: same document, the middle third replaced by other records.
rows, facts = document(1500, 31)
base = [{"role": "system", "content": "Answer exactly.\n\n" + "\n".join(rows)}, ask(sorted(facts)[10])]
call(base)
alt_rows, alt_facts = document(500, 32)
edited = rows[:750] + alt_rows + rows[1250:]
early, late = sorted(facts)[100], sorted(facts)[1400]
partial = []
for key, table in ((early, facts), (late, facts), (sorted(alt_facts)[200], alt_facts)):
    msgs = [{"role": "system", "content": "Answer exactly.\n\n" + "\n".join(edited)}, ask(key)]
    r = call(msgs)
    r["correct"] = str(table[key]) in r["text"]
    partial.append(r)
    print("partial", key, r["ttft"], r["prompt"], r["cached"], r["correct"], flush=True)
out["partial"] = partial

# 4. Long greedy output: cold, then a cache hit on the identical prompt.
rows, facts = document(900, 41)
msgs = [{"role": "user", "content": "\n".join(rows) + "\n\nList the owner team of records 0 through 40, one per line as 'Record N: team-X'."}]
cold = call(msgs, max_tokens=400)
warm = call(msgs, max_tokens=400)
same = 0
for a, b in zip(cold["text"], warm["text"]):
    if a != b:
        break
    same += 1
expected = [f"Record {i}: team-{rows[i].split('owner team-')[1].rstrip('.')}" for i in range(41)]
acc = lambda t: sum(e in t for e in expected)
out["long"] = {"cold": cold, "warm": warm, "common_prefix_chars": same, "cold_right": acc(cold["text"]), "warm_right": acc(warm["text"])}
print("long", cold["cached"], warm["cached"], "identical" if cold["text"] == warm["text"] else f"diverge at char {same}/{len(cold['text'])}",
      "right", acc(cold["text"]), acc(warm["text"]), "of 41", flush=True)

# 5. Thinking on (low effort), multi-turn.
rows, facts = document(1000, 51)
convo = [{"role": "system", "content": "Answer exactly.\n\n" + "\n".join(rows)}]
think = []
for key in random.Random(5).sample(sorted(facts), 3):
    convo.append(ask(key))
    r = call(convo, max_tokens=600, think=True)
    r["correct"] = str(facts[key]) in r["text"]
    think.append(r)
    convo.append({"role": "assistant", "content": r["text"]})
    print("think", r["ttft"], r["prompt"], r["cached"], r["correct"], repr(r["text"][:30]), flush=True)
out["think"] = think

json.dump(out, open(f"pch-{sys.argv[1]}.json", "w"), indent=1)
checks = [r["correct"] for r in out["concurrent"]] + [out["tool"]["correct"]] + [r["correct"] for r in partial] + [r["correct"] for r in think]
print(f"{sys.argv[1]}: {sum(checks)}/{len(checks)} exact checks; long-output {out['long']['cold_right']}/{out['long']['warm_right']} of 41")
