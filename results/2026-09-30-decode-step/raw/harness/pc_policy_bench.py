#!/usr/bin/env python3
"""Prefix-cache policy A/B (ATLAS_GLM_PC_EVICT / ATLAS_GLM_PC_BRANCH): agentic sessions.

Usage: pc_policy_bench.py TAG      (writes pcp-TAG.json; about 3 minutes)
Exit status: 0 done, 3 a request failed or timed out (the caller treats it as NO-GO).

Every session starts with the SAME ~24K-token system prompt (as omp and similar
agent clients do), then its own records table, for 30-48K tokens in total.
Greedy, thinking off; every answer holds an exact 6-digit code, so a wrong
restored state shows up as a wrong answer, and the greedy text is compared
across legs by pc_policy_gate.py.

  P1 new sessions:  s0 turn 1 alone, s1 turn 1 alone, then s2+s3 turn 1 together.
                    ON: s1 finds the fork and plants the branch checkpoint (one
                    extra pass), s2/s3 restore it.
  P2 warm turns:    s0-s3 concurrently, 4 more turns each (turns 2-5).
  P3 churn:         s4 (shared prompt, small table) runs 17 quick turns alone while
                    s0-s3 idle (17 > 16 snapshot slots); then s0-s3 each take turn
                    6 concurrently. Base: s4's tail snapshots evict s0-s3's.
"""
import json, random, statistics, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8893/v1/chat/completions"
TAG = sys.argv[1]
REQ_TIMEOUT_S = 180  # a healthy 48K cold prefill takes well under a minute
FAILED = threading.Event()


def call(messages):
    body = {"model": "glm-5.3-flash-atlas", "messages": messages, "max_tokens": 32, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT_S) as r:
        for raw in r:
            if time.time() - t0 > REQ_TIMEOUT_S:
                raise TimeoutError(f"request exceeded {REQ_TIMEOUT_S}s")
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return (text.strip(), round(ttft or -1, 2), usage.get("prompt_tokens"),
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))


def table(prefix, n, seed):
    rng = random.Random(seed)
    names = [f"{prefix}-{i:05d}" for i in range(n)]
    codes = [rng.randint(100000, 999999) for _ in names]
    rows = [f"Record {i}: {x} has access code {c} and owner team-{rng.randint(1, 99)}."
            for i, (x, c) in enumerate(zip(names, codes))]
    return "\n".join(rows), names, codes


SYS_ROWS, _, _ = table("tool", 1100, 42)  # ~24K tokens, identical in every session
SYSTEM = ("You are a coding agent. The tool registry below is reference material; "
          "answer questions about the user's records exactly.\n\n" + SYS_ROWS)

results, lock = [], threading.Lock()


class Session:
    def __init__(self, s, n_records):
        rows, self.names, self.codes = table(f"unit{s}", n_records, 100 + s)
        self.s = s
        self.rng = random.Random(7 + s)
        self.convo = [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": "My records:\n\n" + rows + "\n\nAcknowledge with OK."},
                      {"role": "assistant", "content": "OK."}]

    def turn(self, phase, turn):
        if FAILED.is_set():
            return
        i = self.rng.randrange(len(self.names))
        self.convo.append({"role": "user",
                           "content": f"What are the access code and the owner team of {self.names[i]}? "
                                      "Answer exactly as: <code>, team-<n>"})
        try:
            text, ttft, prompt, cached = call(self.convo)
        except Exception as e:  # timeout, reset, HTTP error: the leg is NO-GO
            FAILED.set()
            print(f"REQUEST-FAILED {phase} s{self.s} turn {turn}: {e!r}", flush=True)
            return
        near = {self.codes[j] for j in (i - 1, i + 1) if 0 <= j < len(self.codes)}
        r = {"phase": phase, "session": self.s, "turn": turn, "ttft": ttft, "prompt": prompt, "cached": cached,
             "correct": str(self.codes[i]) in text, "off_by_one": any(str(c) in text for c in near), "text": text}
        with lock:
            results.append(r)
            print(phase, "s%d" % self.s, "turn", turn, ttft, prompt, cached, r["correct"], flush=True)
        self.convo.append({"role": "assistant", "content": text})


def together(fns):
    ts = [threading.Thread(target=f) for f in fns]
    [t.start() for t in ts]; [t.join() for t in ts]


t_start = time.time()
S = [Session(s, n) for s, n in enumerate((300, 500, 700, 1000))]  # ~31K .. ~46K tokens
S[0].turn("P1", 1)
S[1].turn("P1", 1)
together([lambda: S[2].turn("P1", 1), lambda: S[3].turn("P1", 1)])
for t in range(2, 6):
    together([lambda x=x: x.turn("P2", t) for x in S])
fast = Session(4, 50)
for t in range(1, 18):
    fast.turn("P3fast", t)
together([lambda x=x: x.turn("P3resume", 6) for x in S])


def med(rows):
    v = [r["ttft"] for r in rows]
    return round(statistics.median(v), 2) if v else None


def pick(ph, f=lambda r: True):
    return [r for r in results if r["phase"] == ph and f(r)]


summary = {
    "tag": TAG, "failed": FAILED.is_set(),
    "correct": sum(r["correct"] for r in results), "n": len(results),
    "off_by_one": sum(r["off_by_one"] and not r["correct"] for r in results),
    "p1_first_session_ttft": med(pick("P1", lambda r: r["session"] == 0)),
    "p1_second_session_ttft": med(pick("P1", lambda r: r["session"] == 1)),
    "p1_later_sessions_ttft": med(pick("P1", lambda r: r["session"] >= 2)),
    "p2_warm_ttft_median": med(pick("P2")),
    "p2_warm_ttft_max": max((r["ttft"] for r in pick("P2")), default=None),
    "p3_fast_ttft_median": med(pick("P3fast")),
    "p3_resume_ttft_median": med(pick("P3resume")),
    "p3_resume_ttft_max": max((r["ttft"] for r in pick("P3resume")), default=None),
    "prompt_tokens_max": max((r["prompt"] or 0 for r in results), default=0),
    "wall_s": round(time.time() - t_start, 1),
}
json.dump({"summary": summary, "results": results}, open(f"pcp-{TAG}.json", "w"), indent=1)
print(json.dumps(summary), flush=True)
sys.exit(3 if FAILED.is_set() else 0)
