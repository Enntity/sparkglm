#!/usr/bin/env python3
"""Probe SparkGLM's behaviour on a short structured call that asks for no thinking."""
import json, sys, time, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8893"
KEY = "sk-local-llm"
MODEL = "glm-5.3-flash-atlas"
SYSTEM = "You are an appraisal step in a larger system. Read the note and record a structured appraisal. Output only the JSON object."
NOTES = [
    "Spent the afternoon fixing the greenhouse door. The hinge was rusted through, so I replaced it with the brass one from the shed. It closes properly now, and the tomatoes should stop getting cold at night.",
    "The delivery arrived two days late and the box was crushed on one corner. Nothing inside was broken, but the manual was missing. I emailed the seller and they answered within an hour with a PDF.",
    "We argued about the route for most of the drive, then found the lake by accident. Nobody said much while the sun went down. On the way back she laughed about the map and I think it is fine now.",
]
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["summary", "mood", "importance", "follow_up"],
          "properties": {"summary": {"type": "string"},
                         "mood": {"type": "string", "enum": ["calm", "pleased", "annoyed", "sad", "mixed"]},
                         "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                         "follow_up": {"type": "boolean"}}}
RF = {"type": "json_schema", "json_schema": {"name": "appraisal", "strict": True, "schema": SCHEMA}}

def post(path, body, timeout=200):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                               headers={"content-type": "application/json", "authorization": "Bearer " + KEY})
    t = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return json.load(f), time.time() - t
    except Exception as e:
        return {"error": repr(e), "body": getattr(e, "read", lambda: b"")()[:300].decode("utf8", "replace")}, time.time() - t

def show(tag, d, dt):
    if "error" in d:
        print(f"{tag}: ERROR {dt:.1f}s {d}"); return
    c = d["choices"][0]; m = c.get("message") or {}
    content = m.get("content") if m else c.get("text")
    reasoning = m.get("reasoning_content") or ""
    u = d.get("usage", {})
    ok = False
    try:
        obj = json.loads(content); ok = set(obj) == set(SCHEMA["required"])
    except Exception:
        obj = None
    print(f"{tag}: {dt:5.1f}s finish={c.get('finish_reason')} completion_tokens={u.get('completion_tokens')} "
          f"reasoning_chars={len(reasoning)} valid_json={ok} content={json.dumps(content)[:230]}")

arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["chat_off", "raw_closed"]
maxtok = int(sys.argv[3]) if len(sys.argv) > 3 else 3072
for i, note in enumerate(NOTES):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Note:\n" + note}]
    if "chat_off" in arms:   # the girls' request shape: thinking asked off
        d, dt = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "temperature": 0.1, "max_tokens": maxtok,
              "response_format": RF, "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}})
        show(f"note{i} chat thinking-off ", d, dt)
    if "chat_low" in arms:   # low effort, no budget
        d, dt = post("/v1/chat/completions", {"model": MODEL, "messages": msgs, "temperature": 0.1, "max_tokens": maxtok,
              "response_format": RF, "reasoning_effort": "low"})
        show(f"note{i} chat effort low   ", d, dt)
    if "raw_closed" in arms: # the proposed fix, by hand: same prompt with the think block closed
        prompt = ("[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>" + SYSTEM + "<|user|>Note:\n" + note
                  + "<|assistant|><think></think>")
        d, dt = post("/v1/completions", {"model": MODEL, "prompt": prompt, "temperature": 0.1, "max_tokens": maxtok,
              "response_format": RF})
        show(f"note{i} raw closed-think  ", d, dt)
