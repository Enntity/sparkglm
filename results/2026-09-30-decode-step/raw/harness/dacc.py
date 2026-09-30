#!/usr/bin/env python3
"""dacc.py ARM | compare BASE ARM [ARM...]        (runs on spark1, standard library only)

Natural-text draft-acceptance set for drafter-context arms. One server start per arm; ARM sends the
set once, in a fixed order, and writes ~/sparkglm-dev/dacc-ARM.json. No forced token counts: every
prompt asks for something that naturally runs past DACC_MAX_TOKENS (default 160).

Blocks (78 requests, about 7 minutes; DACC_QUICK=1 halves it):
  sys   12 base prompts (4 prose, 4 code, 4 qa) x system message {none, short, agent}: the paired
        effect of a system message on the same task. Cold: system variants carry a per-request salt
        so the shared system text is never a prefix-cache hit.
  len   6 tasks over a document of about 300 / 1500 / 4000 / 16000 tokens (no system message):
        below and above the drafter's 2048-token window. Cold.
  turn  6 conversations (3 with the agent system message), three turns each, fed the model's own
        replies: turn 1 cold, turns 2-3 are real warm turns after a prefix-cache restore.

Per request: prompt/completion/cached tokens, finish reason, TTFT, tok/s, and p1, mean_na, tok_step
from the rank-0 scheduler's `Done:` line (container DACC_CONTAINER, default atlas-sparkglm-rank0).

compare: tokens per verify step per group (token-weighted), and for each arm against BASE the paired
per-prompt ratio: median, a bootstrap 90% interval over prompts, and how many prompts went up/down.
A group is UP or DOWN only when the whole interval is on one side of 1; otherwise UNRESOLVED.
Greedy text differs between arms, so single prompts swing about +-10%: read groups, not rows.
"""
import json, os, random, re, statistics, subprocess, sys, time, urllib.request

URL = os.environ.get("DACC_URL", "http://127.0.0.1:8893/v1/chat/completions")
MODEL = "glm-5.3-flash-atlas"
DEV = os.path.expanduser("~/sparkglm-dev")
OUT = os.path.join(DEV, "dacc-{}.json")
CONTAINER = os.environ.get("DACC_CONTAINER", "atlas-sparkglm-rank0")
MAX_TOKENS = int(os.environ.get("DACC_MAX_TOKENS", "160"))
QUICK = os.environ.get("DACC_QUICK") == "1"
DONE = re.compile(r"Done: (\d+) tokens .*?([0-9.]+) tok/s, TTFT=([0-9.]+)ms.*? p1=([0-9.]+) mean_na=([0-9.]+) tok_step=([0-9.]+)")
WORDS = ("amber birch cedar delta ember flint garnet harbor indigo juniper kestrel lumen maple nickel "
         "onyx pewter quartz raven sable topaz umber violet willow zephyr").split()

SYS_SHORT = "You are a helpful assistant. Answer directly and completely."
SYS_AGENT = ("You are a careful coding agent working in a user's repository. Inspect files and run commands before "
             "answering. Work one step at a time, say what you are about to do, and keep explanations short.")
BASE = [  # name, class, user text
    ("prose_hash", "prose", "<prompt omitted: quoted from mmastrac/glm-5.3-flash-4x-gx10 @ 4e63b64, which has no license>"),
    ("prose_tcp", "prose", "Explain what happens when a TCP connection is opened and closed, as a few plain paragraphs."),
    ("prose_rain", "prose", "Describe the water cycle for a ten-year-old, as one long story about a single raindrop."),
    ("prose_vaccine", "prose", "Explain how a vaccine trains the immune system, in plain paragraphs for a general reader."),
    ("code_rbtree", "code", "<prompt omitted: quoted from mmastrac/glm-5.3-flash-4x-gx10 @ 4e63b64, which has no license>"),
    ("code_lru", "code", "Write an LRU cache in Rust with get and put in O(1). Code only, with brief comments."),
    ("code_sql", "code", "Write a SQL query that returns the top 3 customers by total order value per region, with the schema you assume."),
    ("code_go", "code", "Write a Go HTTP server with two JSON endpoints, request logging and graceful shutdown. Code only."),
    ("qa_review", "qa", "List fifteen practical tips for reviewing a pull request, one line each."),
    ("qa_json", "qa", "Return a JSON array of six fictional books, each with title, author, year, genres (array) and a two-sentence summary."),
    ("qa_table", "qa", "Compare TCP, UDP and QUIC in a Markdown table with eight rows, then explain each row in one sentence."),
    ("qa_plan", "qa", "Give a step-by-step plan for migrating a production Postgres database to a new host with no downtime."),
]
LEN_TASKS = [  # name, class, file, instruction
    ("sum_prose", "prose", "nat-prose.txt", "Summarize the text above in detail, in plain paragraphs."),
    ("facts_prose", "qa", "nat-prose.txt", "List the twelve most important points of the text above, one sentence each."),
    ("sum_docs", "prose", "nat-docs.txt", "Summarize the documentation above in detail, in plain paragraphs."),
    ("howto_docs", "qa", "nat-docs.txt", "From the documentation above, write step-by-step instructions for a new user."),
    ("explain_code", "prose", "nat-code.txt", "Explain what the code above does, function by function."),
    ("tests_code", "code", "nat-code.txt", "Write unit tests for the code above. Code only."),
]
LENGTHS = [(300, 1200), (1500, 6000), (4000, 16000), (16000, 62000)]  # nominal tokens, characters
TURNS = [  # name, class, system message or None, first user text
    ("conv_dns", "prose", None, "Explain how DNS resolution works from typing a name to getting an address, in plain paragraphs."),
    ("conv_gc", "prose", None, "Explain how a generational garbage collector works, in plain paragraphs."),
    ("conv_queue", "code", None, "Write a thread-safe bounded queue in Python with blocking put and get. Code, then a short explanation."),
    ("conv_flag", "prose", SYS_AGENT, "I want to add a --dry-run flag to the deploy script. Tell me your plan before touching anything."),
    ("conv_flaky", "prose", SYS_AGENT, "A test passes locally and fails in CI about one run in five. Explain how you would track it down."),
    ("conv_parse", "code", SYS_AGENT, "Write a function that parses a duration like '1h30m15s' into seconds, with tests. Code first."),
]
FOLLOW = ["Now go deeper on the second point you made, in the same style and at the same length.",
          "Give a concrete worked example of that, step by step."]


def salt(i):
    r = random.Random(7919 * (i + 1))
    return "Session " + "-".join(r.sample(WORDS, 3)) + ". "


def build():
    """The request list: dicts with name, block, class, sys, messages (turn rows fill messages later)."""
    reqs = []
    base = BASE[::2] if QUICK else BASE
    for name, cls, text in base:
        for tag, system in (("none", None), ("short", SYS_SHORT), ("agent", SYS_AGENT)):
            msgs = [{"role": "user", "content": text}]
            if system:
                msgs.insert(0, {"role": "system", "content": salt(len(reqs)) + system})
            reqs.append({"name": f"{name}/{tag}", "pair": name, "block": "sys", "class": cls, "sys": tag, "messages": msgs})
    tasks = LEN_TASKS[::2] if QUICK else LEN_TASKS
    for t, (name, cls, fname, inst) in enumerate(tasks):
        doc = open(os.path.join(DEV, fname), errors="replace").read()
        for k, (toks, chars) in enumerate(LENGTHS):
            start = (1777 * (t * len(LENGTHS) + k + 1)) % max(1, len(doc) - chars)
            start = doc.find("\n", start) + 1  # begin at a line start; every request has its own offset
            body = doc[start:start + chars]
            reqs.append({"name": f"{name}/{toks}", "pair": name, "block": "len", "class": cls, "sys": "none", "nominal": toks,
                         "messages": [{"role": "user", "content": "Document:\n\n" + body + "\n\n" + inst}]})
    convs = TURNS[::2] if QUICK else TURNS
    for name, cls, system, text in convs:
        for turn in (1, 2, 3):
            reqs.append({"name": f"{name}/t{turn}", "pair": name, "block": "turn", "class": cls, "sys": "agent" if system else "none",
                         "turn": turn, "system": (salt(len(reqs) - turn + 1) + system) if system else None, "first": text})
    return reqs


def dones():
    o = subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True, errors="replace")
    return DONE.findall(re.sub(r"\x1b\[[0-9;]*m", "", o.stdout + o.stderr))


def stream(messages):
    body = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = {}; fin = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:]); usage = d.get("usage") or usage
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece; fin = c.get("finish_reason") or fin
    n = usage.get("completion_tokens") or 0; wall = time.time() - t0
    return {"prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "ctoks": n, "finish": fin, "ttft": round(ttft or -1, 3),
            "tps": round((n - 1) / (wall - ttft), 2) if ttft and n > 1 and wall > ttft else 0.0, "text": text}


def arm(tag):
    reqs = build(); rows = []; t_start = time.time(); replies = {}
    for i, q in enumerate(reqs):
        if q["block"] == "turn":
            msgs = [{"role": "system", "content": q["system"]}] if q["system"] else []
            msgs.append({"role": "user", "content": q["first"]})
            for k in range(1, q["turn"]):
                msgs += [{"role": "assistant", "content": replies.get((q["pair"], k), "")}, {"role": "user", "content": FOLLOW[k - 1]}]
        else:
            msgs = q["messages"]
        n0 = len(dones())
        try:
            r = stream(msgs)
        except Exception as e:  # keep going: one failed request must not lose the arm
            r = {"prompt": None, "cached": 0, "ctoks": 0, "finish": None, "ttft": -1, "tps": 0.0, "text": "", "error": repr(e)[:200]}
        d = dones(); st = d[n0] if len(d) == n0 + 1 else None  # exactly one new Done line, or no stats
        if q["block"] == "turn":
            replies[(q["pair"], q["turn"])] = r["text"]
        row = {k: v for k, v in q.items() if k not in ("messages", "system", "first")}
        row.update({k: v for k, v in r.items() if k != "text"})
        row.update({"p1": float(st[3]) if st else None, "mean_na": float(st[4]) if st else None,
                    "tok_step": float(st[5]) if st else None, "text": r["text"]})
        rows.append(row)
        json.dump({"arm": tag, "max_tokens": MAX_TOKENS, "quick": QUICK, "elapsed_s": round(time.time() - t_start, 1), "rows": rows},
                  open(OUT.format(tag), "w"), indent=1)
        print(f"{i + 1}/{len(reqs)} {row['name']:22s} prompt={row['prompt']} cached={row['cached']} out={row['ctoks']} "
              f"{row['finish']} tok_step={row['tok_step']} p1={row['p1']} {row['tps']} tok/s", flush=True)
    summary(tag, rows)


def bucket(row):
    p = row.get("prompt") or 0
    return "<33" if p < 33 else "33-200" if p < 200 else "200-2048" if p <= 2048 else ">2048"


GROUPS = [("all", lambda r: True)] + \
    [(f"block={b}", lambda r, b=b: r["block"] == b) for b in ("sys", "len", "turn")] + \
    [(f"sys={s} (block sys)", lambda r, s=s: r["block"] == "sys" and r["sys"] == s) for s in ("none", "short", "agent")] + \
    [(f"prompt {b}", lambda r, b=b: bucket(r) == b) for b in ("<33", "33-200", "200-2048", ">2048")] + \
    [(f"class={c}", lambda r, c=c: r["class"] == c) for c in ("prose", "code", "qa")] + \
    [("cold (cached=0)", lambda r: not r["cached"]), ("warm (cached>0)", lambda r: bool(r["cached"]))]


def usable(r):
    return bool(r.get("tok_step")) and (r.get("ctoks") or 0) >= 48


def agg(rows):
    rows = [r for r in rows if usable(r)]
    toks = sum(r["ctoks"] for r in rows); steps = sum(r["ctoks"] / r["tok_step"] for r in rows)
    return (round(toks / steps, 3) if steps else None, len(rows),
            round(statistics.mean(r["p1"] for r in rows), 3) if rows else None)


def summary(tag, rows):
    bad = [r["name"] for r in rows if not usable(r)]
    print(f"== {tag}: {len(rows)} requests, {sum(r['ctoks'] for r in rows)} tokens; without stats or under 48 tokens: {len(bad)} {bad[:8]}")
    for name, f in GROUPS:
        a = agg([r for r in rows if f(r)])
        if a[1]:
            print(f"{tag} {name:24s} tok/step {a[0]}  n={a[1]}  mean p1 {a[2]}")


def paired(base, other, f):
    b = {r["name"]: r for r in base if f(r) and usable(r)}
    ratios = [r["tok_step"] / b[r["name"]]["tok_step"] for r in other if r["name"] in b and usable(r)]
    if len(ratios) < 4:
        return None
    rng = random.Random(12345); meds = []
    for _ in range(2000):
        meds.append(statistics.median(rng.choices(ratios, k=len(ratios))))
    meds.sort(); lo, hi = meds[100], meds[1899]
    verdict = "UP" if lo > 1.0 else "DOWN" if hi < 1.0 else "UNRESOLVED"
    return statistics.median(ratios), lo, hi, sum(x > 1.005 for x in ratios), sum(x < 0.995 for x in ratios), len(ratios), verdict


def compare(names):
    data = {n: json.load(open(OUT.format(n)))["rows"] for n in names}
    base = names[0]
    print("tokens per verify step (token-weighted) | n")
    print(f"{'group':26s}" + "".join(f"{n:>16s}" for n in names))
    for gname, f in GROUPS:
        cells = [agg([r for r in data[n] if f(r)]) for n in names]
        if any(c[1] for c in cells):
            print(f"{gname:26s}" + "".join(f"{str(c[0]):>10s} n={c[1]:<3d}" for c in cells))
    for n in names[1:]:
        print(f"\n{n} against {base}: paired per-prompt ratio of tok/step (median, 90% interval), prompts up/down")
        for gname, f in GROUPS:
            p = paired(data[base], data[n], f)
            if p:
                print(f"  {gname:24s} {p[0]:.3f} [{p[1]:.3f}, {p[2]:.3f}]  up {p[3]} down {p[4]} of {p[5]}  {p[6]}")
    sys_rows = {n: {r["name"]: r for r in data[n] if r["block"] == "sys" and usable(r)} for n in names}
    print("\nwithin one arm: system message against none, same task (paired ratio median, 90% interval)")
    for n in names:
        for s in ("short", "agent"):
            ratios = [sys_rows[n][k]["tok_step"] / sys_rows[n][k.rsplit("/", 1)[0] + "/none"]["tok_step"]
                      for k in sys_rows[n] if k.endswith("/" + s) and k.rsplit("/", 1)[0] + "/none" in sys_rows[n]]
            if len(ratios) >= 4:
                rng = random.Random(999); meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
                print(f"  {n:12s} {s:6s} {statistics.median(ratios):.3f} [{meds[100]:.3f}, {meds[1899]:.3f}]  n={len(ratios)}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2:])
    elif len(sys.argv) == 2 and sys.argv[1] != "compare":
        arm(sys.argv[1])
    else:
        sys.exit(__doc__)
