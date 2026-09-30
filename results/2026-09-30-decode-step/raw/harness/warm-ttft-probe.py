#!/usr/bin/env python3
"""warm-ttft-probe.py ARM | compare BASE CAND [BASE2] | report ARM   (runs on spark1, stdlib only)

Warm-turn TTFT probe for opt/warm-ttft. ARM runs against the dev pair on :8893 and writes
~/sparkglm-dev/warm-ttft-ARM.json. Every request is temperature 0, thinking off, streamed.
Run every arm with ATLAS_GLM_WARM_TRACE=hash on both ranks (one line per request with a hash of the
prefill's logits; it costs one read of a logits row per request) and save rank 1's log to
~/sparkglm-dev/warm-ttft-ARM.rank1.log when the arm ends.

What one arm runs, in order (a step that would overrun WTP_BUDGET_S, default 285 s, is skipped and
reported; the arm stops at the first request error):

  H0      prompt-logprob hash of a fixed prompt of about 10K tokens (echo + logprobs: a full recompute
          in two chunks and a tail split, bit-reproducible across server starts and binaries).
  round r (2 rounds), for each cached context 2K, 16K, 45K (a registry of 90 / 750 / 2130 salted
          records; salts depend on the round, not the arm). Under ATLAS_GLM_ZERO_ROWS=check, which
          reads the arena back before every zero: one round, 45K first, no prose and no dup.
            T0      cold: the registry + "value of KEY?"                      -> the cold control
            T1..T6  warm: "value of KEY?" (six warm turns, short answers)     -> warm TTFT
            T7      warm: "copy entries a..a+11" (about 250 output tokens)    -> decode, acceptance
            B1..B3  at 45K only: three more lookups while a second request decodes a long
                    answer                                                    -> warm TTFT under load
            N1, N2  warm: a short question answered in about 120 words        -> acceptance on prose
          The history carries the EXPECTED answer of each lookup and copy turn, whatever the model
          said, so a turn's prompt is the same in every arm; N2 alone carries N1's real text.
  dup     round 0 at 2K only: the copy turn's exact request again (its output then runs along tokens
          the conversation has since cached), then N1's exact request: it must still match all of
          its prompt's whole blocks.
  H1      the same hash again, after everything above.

Per turn: client TTFT, engine TTFT (the scheduler's `Done: ... TTFT=`), prompt / cached / completion
tokens, the answer check, p1 / mean_na / tok_step from the `Done:` line, the restore depth and replay
rows from the Marconi line, the `warm-turn` line (spans, logits hash) and the `CHAT_PHASE` tokenize
time when ATLAS_CHAT_PHASE_TIMING=1. Rank-0 lines come from `docker logs atlas-sparkglm-rank0`.

compare BASE CAND [BASE2]: prints the numbers and one line, VERDICT: GO / NO-GO / INCONCLUSIVE.
What CAND must gain is read from the switches its rank 0 ran with (EXPECT below). BASE2, a second run
of the baseline taken after CAND, measures the run-to-run drift the savings must exceed. An arm named
rc* is the release candidate binary (before the finish-cache fix); `compare rc wt-base` judges the
fix. See GO criteria in the hand-off for what each arm is compared with.

report ARM: one arm's numbers, and with the trace the per-step timeline by context.
"""
import hashlib, json, os, random, re, statistics, subprocess, sys, threading, time, urllib.request

BASE = os.environ.get("WTP_URL", "http://127.0.0.1:8893")
URL = BASE + "/v1/chat/completions"
MODEL = "glm-5.3-flash-atlas"
DEV = os.path.expanduser(os.environ.get("WTP_DIR", "~/sparkglm-dev"))
OUT = os.path.join(DEV, "warm-ttft-{}.json")
BUDGET_S = float(os.environ.get("WTP_BUDGET_S", "285"))
CONTEXTS = [("2K", 90), ("16K", 750), ("45K", 2130)]  # label, registry records (about 21 tokens each)
BUSY = "45K-busy"  # the 45K conversation's lookups while a second request decodes
LABELS = [label for label, _ in CONTEXTS] + [BUSY]
WARM_TURNS, BUSY_TURNS = 6, 3
HASH_RECORDS = int(os.environ.get("WTP_HASH_RECORDS", "450"))
BLOCK = 16
ANSI = re.compile(r"\x1b\[[0-9;]*m")
WORDS = ("amber birch cedar delta ember flint garnet harbor indigo juniper kestrel lumen "
         "maple nickel onyx pewter quartz raven sable topaz umber violet willow zephyr").split()
TOPICS = ["why the sky is blue", "how a bicycle stays upright", "what causes ocean tides", "why bread rises",
          "how a vaccine trains the immune system", "why leaves change colour in autumn", "how a compass works",
          "why the moon has phases", "how yeast makes alcohol", "why ice floats on water",
          "how a suspension bridge carries load", "why onions make people cry"]
ESSAY = "Write a detailed essay of about 500 words on the history of the bicycle."

P = {
    "hit": re.compile(r"Marconi SSM cache hit: (\d+) tokens skipped .*?replaying (\d+) SSM tokens to reach (\d+)"),
    "inter": re.compile(r"Marconi intermediate hit: restored from checkpoint at token (\d+) .*?replaying (\d+) SSM tokens to reach (\d+)"),
    "nosnap": re.compile(r"Prefix cache hit: (\d+) tokens .*?no SSM snapshot"),
    "cap": re.compile(r"F83 EP-cache-sync: local_matched=(\d+) agreed_matched=(\d+) \(cap"),
    "done": re.compile(r"Done: (\d+) tokens .*?TTFT=([0-9.]+)ms, serial=([0-9.]+) .*?p1=([0-9.]+) mean_na=([0-9.]+) tok_step=([0-9.]+)"),
    "tokenize": re.compile(r"CHAT_PHASE prepare: .*?template_render_and_tokenize=(\d+)us prompt_tokens=(\d+)"),
}
TRACE = re.compile(r"warm-turn rank=(\d+) slot=\d+ tokens=(\d+) matched=(\d+) restored=(\d+) chunks=(\d+) "
                   r"cached_chunks=(\d+) rows=(\d+) ms: (.*?) wall=([0-9.]+) logits=([0-9a-f]{16})")
ZCHECK_STALE = re.compile(r"ATLAS_GLM_ZERO_ROWS=check: (\w+) holds a nonzero byte at (\d+)")
ZCHECK_CLEAN = "ATLAS_GLM_ZERO_ROWS=check: clean"
BAD = re.compile(r"\bERROR\b|panicked|CUDA error|KV cache exhausted|out of step|environments differ")
SPANS = ("transfer", "zero", "embed", "lookup", "blocks", "meta", "forward", "finish")

# What each switch is derived to save per warm turn, in ms of TTFT (the report has the derivation).
# CAND gets GO on a cell when its paired saving is at least half of this AND outside the noise band.
EXPECT = {
    "ATLAS_GLM_WARM_SKIP_CACHED=1": {"16K": 18.0, "45K": 90.0, BUSY: 90.0},
    "ATLAS_GLM_ZERO_ROWS=1": {"2K": 23.0, "16K": 23.0, "45K": 23.0},
    "ATLAS_GLM_TAIL_CUT_DEEP=1": {"2K": 60.0, "16K": 60.0, "45K": 60.0},
    "ATLAS_GLM_WARM_CHUNK_RUN=1": {BUSY: 200.0},
}
ZERO_SWITCHES = ("ATLAS_GLM_WARM_SKIP_CACHED=1", "ATLAS_GLM_ZERO_ROWS=1")  # judged on spans when both arms sync
NUMERICS = ("ATLAS_GLM_TAIL_CUT_DEEP=1",)  # pass shapes change: logits differ from the baseline's by design
NOISE_MS, NOISE_REL = 8.0, 0.02  # the least a paired median must move to count, before any measured drift
MIN_PAIRS = 8        # paired warm turns a context needs before its TTFT is judged
MIN_HASHED = 12      # same-prompt warm turns with a logits hash in both arms, for the exactness gate
MIN_PROSE = 12       # prose turns per arm before acceptance is judged
ACC_NOGO, ACC_SOFT = 0.03, 0.01  # relative drop of tok_step / mean_na on prose or copy turns


# ---------------------------------------------------------------- prompts and checks

def registry(case, n):
    """Deterministic salted registry: n records `Entry NNNN: key K -> value V`."""
    rng = random.Random(f"warm-ttft-probe/{case}")
    salt = "%012x" % rng.getrandbits(48)
    recs = []
    for i in range(1, n + 1):
        key = f"{rng.choice(WORDS).upper()}-{rng.randrange(1000, 9999)}-{i:04d}"
        val = "-".join(f"{rng.randrange(100):02d}" for _ in range(4))
        recs.append((i, key, val))
    head = (f"Registry {salt}. Below are {n} entries. Each entry has a number, a key and a value.\n"
            "You will be asked to look up values and to copy entries. Copy keys and values exactly.\n\n")
    return head + "\n".join(f"Entry {i:04d}: key {k} -> value {v}" for i, k, v in recs), recs


def ask_value(key):
    return f"Value of {key}? Reply with the value only."


def ask_copy(lo, hi):
    return (f"Copy entries {lo} to {hi} in order, one per line, exactly in this form and nothing else:\n"
            "NNNN. KEY = VALUE")


def ask_prose(topic):
    return f"In about 120 words, explain {topic}."


def copy_text(recs, lo, hi):
    return "\n".join(f"{i:04d}. {k} = {v}" for i, k, v in recs if lo <= i <= hi)


def check_value(text, recs, idx):
    val = recs[idx - 1][2]
    if val in (text or ""):
        return "exact"
    near = [i for i, _, v in recs if v in (text or "")]
    return f"neighbour{near[0] - idx:+d}" if near and abs(near[0] - idx) <= 3 else ("other-record" if near else "wrong")


def check_copy(text, recs, lo, hi):
    want = {k: v for _, k, v in recs}
    got = re.findall(r"([A-Z]+-\d{4}-\d{4})\s*=\s*(\d\d-\d\d-\d\d-\d\d)", text or "")
    right = sum(1 for k, v in got if want.get(k) == v)
    expect = {k for i, k, _ in recs if lo <= i <= hi}
    return {"right": right, "wrong": len(got) - right, "missing": len(expect - {k for k, _ in got})}


# ---------------------------------------------------------------- requests and the log

def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def chat(messages, max_tokens, started=None):
    """One streamed request. `started` (an Event) is set at its first token."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, text, first, last, usage, finish, reasoning = time.time(), "", None, None, {}, None, 0
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ev = json.loads(line[5:])
                usage = ev.get("usage") or usage
                for ch in ev.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    d = ch.get("delta") or {}
                    think = d.get("reasoning_content") or d.get("reasoning") or ""
                    if d.get("content") or think:
                        now = time.time() - t0
                        first, last = first if first is not None else now, now
                        text += d.get("content") or ""
                        reasoning += len(think)
                        if started is not None:
                            started.set()
    except Exception as e:  # a hang or a 500 is a result, not a crash of the probe
        return {"error": repr(e), "text": text, "wall": round(time.time() - t0, 3)}
    finally:
        if started is not None:
            started.set()
    n = usage.get("completion_tokens", 0)
    rate = (n - 1) / (last - first) if first is not None and last > first and n > 1 else 0.0
    return {"text": text, "finish": finish, "ttft": round(first, 4) if first is not None else None,
            "wall": round(time.time() - t0, 3), "rate": round(rate, 2), "reasoning_chars": reasoning,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def logprob_hash():
    """Hash of the prompt logprobs of one fixed prompt (prefill path, full recompute). An error entry
    when the server has no /tokenize or echo+logprobs."""
    doc, recs = registry("hash", HASH_RECORDS)
    msgs = [{"role": "user", "content": doc + "\n\n" + ask_value(recs[40][1])}]
    t0 = time.time()
    try:
        tk = post("/tokenize", {"model": MODEL, "messages": msgs,
                                "chat_template_kwargs": {"thinking": False, "enable_thinking": False}})
        ids = tk.get("tokens") or tk.get("token_ids") or tk.get("ids")
        r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                     "echo": True, "logprobs": 1})
        vals = [x for x in r["choices"][0]["logprobs"]["token_logprobs"] if x is not None]
    except Exception as e:
        return {"error": repr(e), "s": round(time.time() - t0, 2)}
    return {"n": len(vals), "hash": hashlib.sha1(json.dumps(vals).encode()).hexdigest()[:16],
            "sum": round(sum(vals), 4), "s": round(time.time() - t0, 2)}


LOG_POS = [0, 0]
RANK0_LINES = []  # every warm-turn line of rank 0 this arm saw, in order: [prompt tokens, logits hash]
LOG_CMD = (os.environ["WTP_LOG_CMD"].split() if os.environ.get("WTP_LOG_CMD") else
           ["docker", "logs", "--since", str(int(time.time()) - 2), "atlas-sparkglm-rank0"])


def new_log_lines():
    """Rank-0 log lines since the previous call (stdout and stderr tracked apart)."""
    try:
        out = subprocess.run(LOG_CMD, capture_output=True, text=True, errors="replace", timeout=30)
        if out.returncode and "--since" in LOG_CMD:  # a docker without it: the whole log, from now on
            del LOG_CMD[2:4]
            return new_log_lines()
    except (OSError, subprocess.TimeoutExpired):
        return []
    new = []
    for i, stream in enumerate((out.stdout, out.stderr)):
        lines = [ANSI.sub("", l) for l in stream.splitlines()]
        new += lines[LOG_POS[i]:]
        LOG_POS[i] = len(lines)
    RANK0_LINES.extend([tr["tokens"], tr["logits"]] for tr in parse_trace(new) if tr["rank"] == 0)
    return new


def turn_log_lines(tokens_out):
    """Rank-0 lines up to the `Done:` line of the request that produced `tokens_out` tokens (any
    `Done:` line once a second has passed: another request's can come first under load)."""
    lines, t0 = [], time.time()
    while time.time() - t0 < 2.0:
        time.sleep(0.1)
        lines += new_log_lines()
        done = [int(m.group(1)) for m in map(P["done"].search, lines) if m]
        if tokens_out in done or (done and time.time() - t0 > 1.0):
            return lines
    return lines


def parse_trace(lines):
    out = []
    for m in filter(None, map(TRACE.search, lines)):
        spans = {k: float(v) for k, v in (kv.split("=") for kv in m.group(8).split())}
        out.append({"rank": int(m.group(1)), "tokens": int(m.group(2)), "matched": int(m.group(3)),
                    "restored": int(m.group(4)), "chunks": int(m.group(5)), "cached_chunks": int(m.group(6)),
                    "rows": int(m.group(7)), "ms": spans, "wall": float(m.group(9)), "logits": m.group(10)})
    return out


def log_stats(lines):
    s = {k: [[float(g) if "." in g else int(g) for g in m.groups()] for m in map(p.search, lines) if m]
         for k, p in P.items()}
    bad = [l[-220:] for l in lines if BAD.search(l)]
    s["bad"], s["bad_count"] = bad[:5], len(bad)
    s["trace"] = parse_trace(lines)
    s["zcheck_stale"] = [list(m.groups()) for m in map(ZCHECK_STALE.search, lines) if m][:8]
    s["zcheck_clean"] = sum(ZCHECK_CLEAN in l for l in lines)
    return s


def fingerprint():
    """What this arm ran against: the warm-turn and prefix-cache switches rank 0 really has."""
    cmd = ["docker", "exec", "atlas-sparkglm-rank0", "sh", "-c", "cat /proc/[0-9]*/environ 2>/dev/null"]
    try:
        env = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=20).stdout.split("\0")
    except (OSError, subprocess.TimeoutExpired):
        env = []
    if os.environ.get("WTP_ENV") is not None:  # testing off the pair
        env = os.environ["WTP_ENV"].split()
    keep = ("ATLAS_GLM_WARM_", "ATLAS_GLM_ZERO_ROWS", "ATLAS_GLM_TAIL_CUT", "ATLAS_GLM_PC_", "ATLAS_MARCONI_",
            "ATLAS_NO_TAIL_SPLIT", "ATLAS_PREFIX_SUBBLOCK", "ATLAS_CHAT_PHASE_TIMING", "ATLAS_PROFILE_PREFILL",
            "ATLAS_DFLASH_FIRST_APPEND", "ATLAS_MAX_BATCH_TOKENS")
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename, "url": URL, "model": MODEL,
            "request": "temperature 0, top_p 1, stream, thinking off",
            "prompt_class": "salted registry lookups and 12-entry copies with the expected answers as history, "
                            "two 120-word prose turns per conversation, one 45K conversation under a second stream",
            "env": sorted({e for e in env if e.startswith(keep)})}


# ---------------------------------------------------------------- the arm

class Stop(Exception):
    """A request failed: the arm ends (a desynced pair would time every later request out)."""


def turn(msgs, user, max_tokens, kind, history=None):
    """Send `msgs` + `user`; append the user message and `history` (or the model's text) to `msgs`."""
    msgs.append({"role": "user", "content": user})
    r = chat(msgs, max_tokens)
    msgs.append({"role": "assistant", "content": history if history is not None else r.get("text", "")})
    r["kind"] = kind
    r["log"] = log_stats(turn_log_lines(r.get("completion_tokens")))
    return r


def busy_turns(conv, msgs, recs, rng):
    """Three lookups of this conversation while a second request decodes a long answer."""
    started, box = threading.Event(), {}
    bg = threading.Thread(target=lambda: box.update(chat([{"role": "user", "content": ESSAY}], 600, started)))
    bg.start()
    started.wait(30)
    for _ in range(BUSY_TURNS):
        idx = rng.randrange(1, len(recs) + 1)
        t = turn(msgs, ask_value(recs[idx - 1][1]), 96, "busy", recs[idx - 1][2])
        t["asked"], t["check"] = idx, check_value(t.get("text"), recs, idx)
        t["bg_running"] = bg.is_alive()
        conv["turns"].append(t)
        if "error" in t:
            break
    bg.join(150)
    conv["background"] = {k: box.get(k) for k in ("rate", "completion_tokens", "ttft", "wall", "error")}
    new_log_lines()  # the background request's own lines


def conversation(label, n_records, rnd, extras):
    case = f"{label}/r{rnd}"
    doc, recs = registry(case, n_records)
    rng = random.Random(f"warm-ttft-probe/q/{case}")
    conv, msgs = {"ctx": label, "round": rnd, "case": case, "records": n_records, "turns": []}, []

    def add(t):
        conv["turns"].append(t)
        if "error" in t:
            raise Stop(conv)

    try:
        for i in range(WARM_TURNS + 1):
            idx = rng.randrange(1, n_records + 1)
            # Room for the low-effort reasoning GLM-5 emits with thinking off; the answer is about 12 tokens.
            t = turn(msgs, (doc + "\n\n" if i == 0 else "") + ask_value(recs[idx - 1][1]), 96,
                     "cold" if i == 0 else "warm", recs[idx - 1][2])
            t["asked"], t["check"] = idx, check_value(t.get("text"), recs, idx)
            add(t)
        lo = rng.randrange(1, n_records - 12)
        want = copy_text(recs, lo, lo + 11)
        copy_request = msgs + [{"role": "user", "content": ask_copy(lo, lo + 11)}]
        t = turn(msgs, ask_copy(lo, lo + 11), 480, "copy", want)
        t["check"] = check_copy(t.get("text"), recs, lo, lo + 11)
        add(t)
        if "busy" in extras:
            busy_turns(conv, msgs, recs, rng)
            if "error" in conv["turns"][-1]:
                raise Stop(conv)
        topics = TOPICS[(2 * CONTEXTS.index((label, n_records)) + 6 * rnd) % len(TOPICS):][:2]
        topics = [] if "no-prose" in extras else topics
        prose_request = msgs + [{"role": "user", "content": ask_prose(TOPICS[0] if not topics else topics[0])}]
        for topic in topics:
            add(turn(msgs, ask_prose(topic), 160, "prose"))
        if "dup" in extras:
            # The copy turn again: at temperature 0 it decodes along tokens the conversation's later
            # prompts have cached. A rank that then gave back references it never took would cut the
            # match of every later turn there; N1's request again shows whether it did.
            d1 = chat(copy_request, 480)
            d1["log"] = log_stats(turn_log_lines(d1.get("completion_tokens")))
            d2 = chat(prose_request, 8)
            d2["log"] = log_stats(turn_log_lines(d2.get("completion_tokens")))
            conv["dup"] = {"armed": (d1.get("text") or "").strip() == want and not d1.get("reasoning_chars"),
                           "copy_error": d1.get("error"), "error": d2.get("error"),
                           "cached": d2.get("cached_tokens"), "prompt": d2.get("prompt_tokens"),
                           "caps": len(d1["log"]["cap"]) + len(d2["log"]["cap"])}
            if "error" in d1 or "error" in d2:
                raise Stop(conv)
    except Stop:
        conv["stopped"] = True
    return conv


def show(c):
    print(f"{c['case']}: " + " | ".join(
        f"{t['kind']} ttft={t.get('ttft')} cached={t.get('cached_tokens')}/{t.get('prompt_tokens')} "
        f"out={t.get('completion_tokens')} {t.get('check', '')}" if "error" not in t else f"ERROR {t['error']}"
        for t in c["turns"]) + (f" | dup {c['dup']}" if "dup" in c else ""), flush=True)


EST = {"2K": 22, "16K": 28, "45K": 42}  # seconds one conversation takes; raised to what this arm measures
EXTRA_S = {"busy": 18, "dup": 10}
CHECK_SLOW = {"2K": 2.0, "16K": 2.0, "45K": 3.0}  # ZERO_ROWS=check reads about 3 GB back before every zero


def arm(name):
    t0, fp = time.time(), fingerprint()
    checking = "ATLAS_GLM_ZERO_ROWS=check" in fp["env"]
    rounds = int(os.environ.get("WTP_ROUNDS", "1" if checking else "2"))
    contexts = CONTEXTS[::-1] if checking else CONTEXTS
    first = chat([{"role": "user", "content": "Reply with the single word: ready"}], 8)  # graph warm-up, not scored
    new_log_lines()
    out = {"arm": name, "fingerprint": fp, "rounds": rounds, "convs": [], "skipped": [], "hashes": [],
           "stopped": first.get("error")}
    if not out["stopped"]:
        out["hashes"].append(logprob_hash())
        print("H0", out["hashes"][0], flush=True)
        time.sleep(0.3)
        new_log_lines()
    hash_s = 1.2 * ((out["hashes"] or [{}])[0].get("s") or 10) + 3  # kept free for H1
    for rnd in range(rounds):
        for label, n in contexts:
            extras = ["busy"] * (label == "45K") + ["dup"] * (label == "2K" and rnd == 0 and not checking)
            extras += ["no-prose"] * checking
            slow = CHECK_SLOW[label] if checking else 1.0
            need = slow * (EST[label] + sum(EXTRA_S.get(x, 0) for x in extras))
            if out["stopped"] or time.time() - t0 + need + hash_s > BUDGET_S:
                out["skipped"].append(f"{label}/r{rnd}")
                print(f"{label}/r{rnd}: skipped ({'a request failed' if out['stopped'] else 'time'})", flush=True)
                continue
            t1 = time.time()
            c = conversation(label, n, rnd, extras)
            EST[label] = max(EST[label], 1.1 * (time.time() - t1) / slow - sum(EXTRA_S.get(x, 0) for x in extras))
            out["convs"].append(c)
            show(c)
            if c.get("stopped"):
                out["stopped"] = f"a request of {c['case']} failed"
    if not out["stopped"]:
        out["hashes"].append(logprob_hash())
        print("H1", out["hashes"][-1], flush=True)
    time.sleep(0.5)
    out["tail_log"] = log_stats(new_log_lines())
    out["rank0_lines"] = RANK0_LINES
    out["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(DEV, exist_ok=True)
    json.dump(out, open(OUT.format(name), "w"), indent=1)
    s = summary(out)
    print(json.dumps({"arm": name, "elapsed_s": out["elapsed_s"], "stopped": out["stopped"], "skipped": out["skipped"],
                      **{k: s[k] for k in ("errors", "answers_exact", "answers_total", "copy", "hashes", "caps",
                                           "engine_ttft", "cold_ttft", "bad_count", "zcheck", "dup", "hashed")}}))
    print(f"now save rank 1's log:  docker logs atlas-sparkglm-rank1 > {DEV}/warm-ttft-{name}.rank1.log 2>&1  (on its host)")


# ---------------------------------------------------------------- summary

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def ms(x):
    return None if x is None else round(1000 * x, 1)


def share(flags):
    return round(sum(flags) / len(flags), 3) if flags else None


def restore_depth(log):
    """Tokens restored from an SSM snapshot on this turn (0: none) and the rows replayed after it."""
    for k in ("inter", "hit"):
        if log.get(k):
            a, replay, total = log[k][0]
            return (a if k == "inter" else total - replay), replay
    return 0, None


def done_of(t):
    """The `Done:` line of this turn's request: by its token count, else the only one."""
    done = (t.get("log") or {}).get("done") or []
    mine = [d for d in done if d[0] == t.get("completion_tokens")] or (done if len(done) == 1 else [])
    return (mine or [[None] * 6])[-1]


def trace_of(t):
    """Rank 0's warm-turn line of this turn's request: by its prompt length."""
    mine = [tr for tr in (t.get("log") or {}).get("trace") or []
            if tr["rank"] == 0 and tr["tokens"] == t.get("prompt_tokens")]
    return mine[-1] if mine else None


def label_of(c, t):
    return BUSY if t["kind"] == "busy" else c["ctx"]


def summary(d):
    turns = [(c, f"{c['case']}:{i}", t) for c in d["convs"] for i, t in enumerate(c["turns"])]
    ok = [(c, key, t) for c, key, t in turns if "error" not in t]
    warm = [(c, key, t) for c, key, t in ok if t["kind"] != "cold"]
    logs = [t.get("log") or {} for _, _, t in turns] + [d.get("tail_log") or {}]
    by = lambda kinds: {label: [t for c, _, t in ok if label_of(c, t) == label and t["kind"] in kinds]
                        for label in LABELS}
    lookups, cold = by({"warm", "busy"}), by({"cold"})
    acc = lambda kind: {"n": len(ts := [t for _, _, t in ok if t["kind"] == kind]),
                        "rate": med([t.get("rate") for t in ts]), "tok_step": med([done_of(t)[5] for t in ts]),
                        "p1": med([done_of(t)[3] for t in ts]), "mean_na": med([done_of(t)[4] for t in ts])}
    answers = {key: t["check"] for _, key, t in turns if isinstance(t.get("check"), str)}
    copies = [t["check"] for _, _, t in ok if isinstance(t.get("check"), dict)]
    timeline = {}
    for label, ts in lookups.items():
        trs = [tr for tr in map(trace_of, ts) if tr]
        if trs:
            timeline[label] = {**{k: med([tr["ms"].get(k) for tr in trs]) for k in SPANS},
                               **{k: med([tr[k] for tr in trs]) for k in ("wall", "chunks", "cached_chunks", "rows")},
                               "between_chunks": med([tr["wall"] - sum(tr["ms"].values()) for tr in trs]), "n": len(trs)}
    tokenize = lambda t: next((us / 1e3 for us, n in (t.get("log") or {}).get("tokenize") or []
                               if n == t.get("prompt_tokens")), None)
    prev = {key: c["turns"][i - 1].get("prompt_tokens") or 0 for c in d["convs"] for i, t in enumerate(c["turns"])
            for key in [f"{c['case']}:{i}"] if i >= 1}
    dups = [c["dup"] for c in d["convs"] if "dup" in c]
    return {
        "env": d["fingerprint"].get("env", []),
        "stopped": d.get("stopped"),
        "errors": [key for _, key, t in turns if "error" in t],
        "skipped": d.get("skipped") or [],
        "answers": answers, "answers_exact": sum(v == "exact" for v in answers.values()), "answers_total": len(answers),
        "copy": {k: sum(c[k] for c in copies) for k in ("right", "wrong", "missing")},
        "hashes": [h.get("hash") for h in d.get("hashes") or []],
        "hash_s": [h.get("s") for h in d.get("hashes") or []],
        # Per turn, for pairing two arms: what the request was and what it cost.
        "turn": {key: {"label": label_of(c, t), "kind": t["kind"], "text": t.get("text"), "ttft": ms(t.get("ttft")),
                       "engine": done_of(t)[1], "cached": t.get("cached_tokens"), "prompt": t.get("prompt_tokens"),
                       "restored": restore_depth(t.get("log") or {})[0], "replay": restore_depth(t.get("log") or {})[1],
                       "logits": (trace_of(t) or {}).get("logits"), "spans": (trace_of(t) or {}).get("ms"),
                       "bg_running": t.get("bg_running")}
                 for c, key, t in ok},
        "warm_ttft": {k: {"n": len(v), "median": med([t["ttft"] for t in v]),
                          "max": max([t["ttft"] for t in v if t["ttft"] is not None], default=None)}
                      for k, v in lookups.items()},
        "engine_ttft": {k: med([done_of(t)[1] for t in v]) for k, v in lookups.items()},
        "cold_ttft": {k: med([t["ttft"] for t in v]) for k, v in cold.items()},
        "prompt_tokens": {k: med([t.get("prompt_tokens") for t in v]) for k, v in lookups.items()},
        "replay_rows": {k: med([restore_depth(t.get("log") or {})[1] for t in v]) for k, v in lookups.items()},
        # The share of warm turns whose match reached the last whole block of the previous turn's prompt
        # (the template reproduced that prompt to its end, which the deep cut needs).
        "match_at_prev_end": share([t.get("cached_tokens") == prev[key] // BLOCK * BLOCK
                                    for _, key, t in warm if key in prev and t["kind"] in ("warm", "copy", "busy")]),
        "nosnap": sum(len(l.get("nosnap") or []) for l in logs),
        "caps": sum(len(l.get("cap") or []) for l in logs) + sum(x.get("caps") or 0 for x in dups),
        "prose": acc("prose"), "copy_acc": acc("copy"),
        "background": [c["background"] for c in d["convs"] if "background" in c],
        "busy_overlap": sum(bool(t.get("bg_running")) for _, _, t in ok if t["kind"] == "busy"),
        "dup": dups,
        "tokenize_ms": {k: med([tokenize(t) for t in v]) for k, v in lookups.items()},
        "hashed": sum(bool((trace_of(t) or {}).get("logits")) for _, _, t in warm),
        "trace_synced": "ATLAS_GLM_WARM_TRACE=1" in d["fingerprint"].get("env", []),
        "timeline": timeline,
        "bad_log": [b for l in logs for b in l.get("bad") or []][:6],
        "bad_count": sum(l.get("bad_count") or 0 for l in logs),
        "zcheck": {"clean": sum(l.get("zcheck_clean") or 0 for l in logs),
                   "stale": [z for l in logs for z in l.get("zcheck_stale") or []][:8]},
        "rank0_lines": d.get("rank0_lines") or [],
    }


def rank1(arm_name):
    cands = [f"{DEV}/warm-ttft-{arm_name}.rank1.log", f"{DEV}/{arm_name}.rank1.log"]
    path = next((p for p in cands if os.path.exists(p)), None)
    if not path:
        return None
    s = log_stats([ANSI.sub("", l) for l in open(path, errors="replace").read().splitlines()])
    return {"path": path, "bad_count": s["bad_count"], "bad": s["bad"],
            "lines": [[tr["tokens"], tr["logits"]] for tr in s["trace"] if tr["rank"] == 1],
            "zcheck_stale": s["zcheck_stale"], "zcheck_clean": s["zcheck_clean"]}


def show_arm(name, S):
    print(f"{name}: env={S['env']}")
    print(f"{name}: stopped={S['stopped']} errors={S['errors']} skipped={S['skipped']} bad_count={S['bad_count']} "
          f"{S['bad_log']} | no-snapshot hits {S['nosnap']} | F83 caps {S['caps']} | prompt-logprob hash [start, end] "
          f"{S['hashes']} ({S['hash_s']} s) | answers exact {S['answers_exact']}/{S['answers_total']} | copied pairs "
          f"{S['copy']} | warm turns with a logits hash {S['hashed']} | match at the previous prompt's last block "
          f"{S['match_at_prev_end']}")
    for label in LABELS:
        w = S["warm_ttft"][label]
        if not w["n"]:
            continue
        print(f"{name} {label} (~{S['prompt_tokens'][label]} prompt tokens, replay {S['replay_rows'][label]} rows): "
              f"warm TTFT ms engine median {S['engine_ttft'][label]}, client median {ms(w['median'])} max {ms(w['max'])} "
              f"n {w['n']}; template+tokenize {S['tokenize_ms'][label]}; cold client TTFT {ms(S['cold_ttft'].get(label))}")
    for kind in ("prose", "copy_acc"):
        a = S[kind]
        print(f"{name} {kind.split('_')[0]} turns (n {a['n']}): tok/s {a['rate']} tok_step {a['tok_step']} p1 {a['p1']} "
              f"mean_na {a['mean_na']}")
    for label, row in S["timeline"].items():
        print(f"{name} timeline {label} (rank 0, median ms of {row['n']} lines; "
              f"{'synced spans' if S['trace_synced'] else 'host-time spans'}): "
              + " ".join(f"{k}={row[k]}" for k in SPANS) + f" between_chunks={round(row['between_chunks'], 1)} "
              f"wall={row['wall']} | chunks={row['chunks']} cached_chunks={row['cached_chunks']} rows={row['rows']}")
    if S["background"]:
        print(f"{name} busy case: second stream {S['background']}, lookups that overlapped it {S['busy_overlap']}")
    if S["dup"]:
        print(f"{name} duplicate turn: {S['dup']}")
    if S["zcheck"]["clean"] or S["zcheck"]["stale"]:
        print(f"{name} ZERO_ROWS check (rank 0): {S['zcheck']['clean']} clean passes, stale {S['zcheck']['stale']}")


def report(name):
    show_arm(name, summary(json.load(open(OUT.format(name)))))
    print(f"rank 1: {json.dumps({k: v for k, v in (rank1(name) or {}).items() if k != 'lines'}) or 'no saved log'}")


# ---------------------------------------------------------------- verdict

def same_prompts(A, B):
    """Turns whose request is the same in both arms: every turn but a conversation's second prose
    turn, whose history holds the first one's real text."""
    same = set()
    for key, a in A["turn"].items():
        b = B["turn"].get(key)
        if b is None or a["prompt"] != b["prompt"] or a["kind"] == "cold":
            continue
        case, i = key.rsplit(":", 1)
        before = A["turn"].get(f"{case}:{int(i) - 1}"), B["turn"].get(f"{case}:{int(i) - 1}")
        if a["kind"] == "prose" and before[0] and before[0]["kind"] == "prose" and before[0]["text"] != before[1]["text"]:
            continue
        same.add(key)
    return same


def paired(A, B, same, field="engine"):
    """Per label, the paired savings A - B in ms over the same-prompt warm turns."""
    out = {}
    for label in LABELS:
        keys = [k for k in sorted(same) if A["turn"][k]["label"] == label
                and A["turn"][k][field] is not None and B["turn"][k][field] is not None]
        d = [A["turn"][k][field] - B["turn"][k][field] for k in keys]
        base = med([A["turn"][k][field] for k in keys])
        out[label] = {"n": len(d), "saved": med(d), "better": round(sum(x > 0 for x in d) / len(d), 2) if d else None,
                      "worse": round(sum(x < 0 for x in d) / len(d), 2) if d else None, "base": base}
    return out


def span_saving(A, B, same, label):
    """Median zero + embed device time saved per warm turn, when both arms sync their spans."""
    d = [sum(A["turn"][k]["spans"][s] - B["turn"][k]["spans"][s] for s in ("zero", "embed")) for k in same
         if A["turn"][k]["label"] == label and A["turn"][k]["spans"] and B["turn"][k]["spans"]]
    return med(d), len(d)


def rel_drop(a, b):
    return None if not a or b is None else (a - b) / a


def compare(a, b, a2=None):
    load = lambda x: summary(json.load(open(OUT.format(x))))
    A, B = load(a), load(b)
    A2 = load(a2) if a2 else None
    for name, S in ((a, A), (b, B)) + (((a2, A2),) if a2 else ()):
        show_arm(name, S)
    print(f"== {a} (baseline) vs {b}" + (f", drift from {a2}" if a2 else ""))
    R1 = rank1(b)
    print(f"rank 1 ({b}): " + (json.dumps({k: v for k, v in R1.items() if k != "lines"}) if R1 else "no saved log"))
    why, soft = [], []
    new = [s for s in EXPECT if s in B["env"] and s not in A["env"]]
    numerics = [s for s in NUMERICS if s in new]
    checking = "ATLAS_GLM_ZERO_ROWS=check" in B["env"]
    synced = A["trace_synced"] and B["trace_synced"]
    diagnostic = checking or B["trace_synced"] != A["trace_synced"]  # time means nothing
    rc = a.startswith("rc")  # the baseline is the binary without the finish-cache fix

    # -- it ran, and what it ran
    if B["stopped"] or B["errors"] or A["stopped"] or A["errors"]:
        why.append(f"request errors ({a}: {A['stopped'] or A['errors'] or 'none'}; {b}: {B['stopped'] or B['errors']})")
    if B["bad_count"] - len(B["zcheck"]["stale"]) > A["bad_count"]:
        why.append(f"new error lines in {b}'s rank-0 log")
    if R1 and R1["bad_count"] > len(R1["zcheck_stale"]):
        why.append(f"error lines in {b}'s rank-1 log")
    if not new and not rc and not checking and not diagnostic:
        soft.append(f"{b} runs no warm-turn switch that {a} lacks")

    # -- answers
    flips = sorted(k for k, v in B["answers"].items() if v != "exact" and A["answers"].get(k) == "exact")
    wrong_pairs = max(0, B["copy"]["wrong"] + B["copy"]["missing"] - A["copy"]["wrong"] - A["copy"]["missing"])
    print(f"answers exact in {a} only: {flips or 'none'}; copied pairs lost: {wrong_pairs}")
    if len(flips) + wrong_pairs >= 2:
        why.append(f"{len(flips) + wrong_pairs} answers or copied pairs regressed")
    elif len(flips) + wrong_pairs == 1:
        soft.append("one flipped answer or pair: repeat both arms")

    # -- cold prefill, bit for bit
    ha, hb = [h for h in A["hashes"] if h], [h for h in B["hashes"] if h]
    rc_hash = os.environ.get("WTP_RC_HASH")
    if len(set(hb)) > 1 or (not numerics and len(set(ha + hb)) > 1) or (rc_hash and set(ha) - {rc_hash}):
        why.append("prompt-logprob hash differs (a cold prefill is not bit-identical)")
    elif len(ha) < 2 or len(hb) < 2:
        soft.append("prompt-logprob hash missing in an arm (cold exactness not checked)")

    # -- warm prefill, bit for bit: the logits of the same request in both arms
    same = same_prompts(A, B)
    hashed = sorted(k for k in same if A["turn"][k]["logits"] and B["turn"][k]["logits"])
    differ = [k for k in hashed if A["turn"][k]["logits"] != B["turn"][k]["logits"]]
    print(f"same-prompt warm turns: {len(same)}; with a logits hash in both arms: {len(hashed)}; hashes differ on "
          f"{len(differ)} {differ[:4]}")
    if numerics:
        print(f"{b} changes pass shapes ({', '.join(numerics)}): its logits are not compared with {a}'s")
    elif differ:
        why.append(f"warm-turn logits differ on {len(differ)} of {len(hashed)} same-prompt turns (not exact)")
    elif len(hashed) < MIN_HASHED and not rc:
        soft.append(f"only {len(hashed)} same-prompt warm turns carry a logits hash in both arms (need {MIN_HASHED}: "
                    "run both with ATLAS_GLM_WARM_TRACE=hash)")
    if R1 and not numerics and hashed:
        # Rank 1's lines line up with rank 0's by order; where rank 0's agree between the arms, so must rank 1's.
        R1a = rank1(a)
        tails = [(S["rank0_lines"], r["lines"][-len(S["rank0_lines"]):] if S["rank0_lines"] else [])
                 for S, r in ((A, R1a), (B, R1)) if r]
        if len(tails) == 2 and all([x[0] for x in r0] == [x[0] for x in r1] for r0, r1 in tails):
            (a0, a1), (b0, b1) = tails
            bad = [i for i in range(min(len(a0), len(b0))) if a0[i] == b0[i] and a1[i] != b1[i]]
            print(f"rank 1 logits: {len(bad)} differ where rank 0's agree, of {min(len(a0), len(b0))} requests")
            if bad:
                why.append("rank 1's warm-turn logits differ between the arms where rank 0's do not")
        else:
            soft.append("rank 1's warm-turn lines do not line up with rank 0's in both arms: its logits are unchecked")
    elif not R1:
        soft.append(f"rank 1's log of {b} was not saved: the worker is unchecked")

    # -- cache behaviour
    diff = lambda f: sorted(k for k in same if A["turn"][k][f] != B["turn"][k][f])
    cached_diff, restored_diff = diff("cached"), diff("restored")
    print(f"cached_tokens differ on {len(cached_diff)} {cached_diff[:4]}; restore depth differs on "
          f"{len(restored_diff)} {restored_diff[:4]}")
    if B["caps"]:
        why.append(f"{B['caps']} F83 cap lines in {b}: the ranks matched different prefixes")
    if numerics:
        shallow = [k for k in restored_diff if B["turn"][k]["restored"] < A["turn"][k]["restored"]]
        if cached_diff or shallow or B["nosnap"] > A["nosnap"]:
            why.append(f"{b} matched less or restored shallower than {a} {(cached_diff + shallow)[:3]}")
        for label in LABELS:
            ra, rb = A["replay_rows"][label], B["replay_rows"][label]
            if ra is not None and rb is not None and rb > ra - BLOCK / 2:
                soft.append(f"replay at {label} is {rb} rows against {ra}: not the block fewer the deeper cut gives")
        if B["match_at_prev_end"] is not None and B["match_at_prev_end"] < 1:
            soft.append("some warm turns did not match to the previous prompt's last block: the deep checkpoint "
                        "was out of reach there")
    elif not rc and (cached_diff or restored_diff or B["nosnap"] > A["nosnap"]):
        why.append(f"{b} matched or restored differently (cache behaviour changed)")
    for S, name in ((B, b),) + (((A, a),) if not rc else ()):
        for dup in S["dup"]:
            want = (dup["prompt"] or 0) // BLOCK * BLOCK
            if not dup["armed"]:
                print(f"{name} duplicate turn: the copy was not reproduced token for token, nothing was tested")
            elif dup["cached"] != want and name == b:
                why.append(f"after a duplicated turn the next one matched {dup['cached']} of {want} tokens")
    if rc:
        print(f"{a} is the release candidate: its F83 caps ({A['caps']}) and duplicate turn ({A['dup']}) are what "
              f"the fix removes; {b} must show none")
        if not A["caps"]:
            soft.append(f"{a} logged no F83 cap either: this run did not exercise what the fix changes")

    # -- ZERO_ROWS=check
    if checking:
        c0, c1 = B["zcheck"]["clean"], (R1 or {}).get("zcheck_clean", 0)
        if B["zcheck"]["stale"] or (R1 and R1["zcheck_stale"]):
            why.append("ZERO_ROWS check found bytes a trimmed zero would leave")
        elif not c0:
            soft.append("ZERO_ROWS=check logged no pass on rank 0 (is RUST_LOG at info?)")
        elif not R1:
            soft.append("ZERO_ROWS=check: rank 1's log was not saved, its arena is unchecked")
        elif c1 < 0.95 * c0:
            soft.append(f"ZERO_ROWS=check: rank 1 logged {c1} clean passes against rank 0's {c0}: its log does not "
                        "cover the arm, its arena is unchecked")
        print(f"ZERO_ROWS check: clean passes rank 0 {c0}, rank 1 {c1}. A clean check qualifies this workload only: "
              f"contexts to 45K, at most two streams, this arena size ({[e for e in B['env'] if 'BATCH_TOKENS' in e] or 'default'})")

    # -- time
    pairs = paired(A, B, same)
    drift = paired(A, A2, same_prompts(A, A2) & same) if A2 else {}
    expect = {}
    for s in new:
        for label, gain in EXPECT[s].items():
            expect[label] = expect.get(label, 0.0) + gain
    for label in LABELS:
        p = pairs[label]
        if not p["n"]:
            continue
        dr = (drift.get(label) or {}).get("saved")
        band = max(NOISE_MS, NOISE_REL * (p["base"] or 0), 2 * abs(dr) if dr is not None else 0)
        gain = expect.get(label)
        print(f"{label}: engine TTFT paired saving median {p['saved']:+.1f} ms over {p['n']} turns ({p['better']} better, "
              f"{p['worse']} worse; baseline median {p['base']}); noise band {band:.1f} ms"
              + (f" (drift {a} to {a2} {dr:+.1f})" if dr is not None else "")
              + (f"; derived saving {gain:.0f}" if gain else ""))
        if diagnostic:
            continue
        if p["n"] < (BUSY_TURNS if label == BUSY else MIN_PAIRS):
            soft.append(f"{label}: only {p['n']} paired warm turns")
            continue
        if p["saved"] < -band and p["worse"] >= 0.75:
            why.append(f"warm TTFT at {label} is {-p['saved']:.1f} ms worse, beyond the noise band")
        if not gain:
            continue
        if synced and all(s in ZERO_SWITCHES for s in new):
            continue  # judged on the spans below
        if gain < band:
            soft.append(f"{label}: the derived {gain:.0f} ms is inside the {band:.1f} ms noise band, TTFT cannot show "
                        "it (compare two ATLAS_GLM_WARM_TRACE=1 arms to judge it on the spans)")
        elif p["saved"] < max(gain / 2, band) or p["better"] < 0.75:
            soft.append(f"{label}: saved {p['saved']:.1f} ms ({p['better']} of turns better); GO needs "
                        f"{max(gain / 2, band):.1f} ms and 0.75")
    if synced and new and all(s in ZERO_SWITCHES for s in new):
        for label, gain in expect.items():
            saved, n = span_saving(A, B, same, label)
            if saved is None:
                continue
            print(f"{label}: zero+embed device time saved {saved:+.1f} ms over {n} turns (derived {gain:.0f})")
            if n < MIN_PAIRS and label != BUSY:
                soft.append(f"{label}: only {n} paired traced turns")
            elif saved < gain / 2:
                soft.append(f"{label}: zero+embed saved {saved:.1f} ms; GO needs {gain / 2:.1f}")
    if BUSY in expect and A["busy_overlap"] + B["busy_overlap"] < 2 * BUSY_TURNS:
        soft.append("the second stream had finished before some busy lookups: the case under load is incomplete")
    cold = [(l, A["cold_ttft"].get(l), B["cold_ttft"].get(l)) for l, _ in CONTEXTS]
    for label, x, y in cold:
        if x and y and y > 1.05 * x + 0.05:
            soft.append(f"cold TTFT at {label} is {ms(y)} against {ms(x)} ms (one or two samples: repeat)")

    # -- decode: acceptance on prose and copy turns
    for kind, name in (("prose", "prose"), ("copy_acc", "copy")):
        x, y = A[kind], B[kind]
        drops = {m: rel_drop(x[m], y[m]) for m in ("tok_step", "mean_na", "rate")}
        noise = {m: abs(rel_drop(x[m], A2[kind][m]) or 0) for m in drops} if A2 else {}
        print(f"{name} turns ({x['n']}/{y['n']}): tok_step {x['tok_step']} -> {y['tok_step']}, mean_na {x['mean_na']} -> "
              f"{y['mean_na']}, p1 {x['p1']} -> {y['p1']}, tok/s {x['rate']} -> {y['rate']}"
              + (f"; run-to-run {({m: round(v, 3) for m, v in noise.items()})}" if noise else ""))
        if diagnostic:
            continue
        for m in ("tok_step", "mean_na"):
            d = drops[m]
            if d is None:
                if numerics:
                    soft.append(f"no {m} for the {name} turns (acceptance not checked)")
                continue
            if numerics:
                nogo, maybe = max(ACC_NOGO, 2 * noise.get(m, 0)), max(ACC_SOFT, noise.get(m, 0))
                if d > nogo:
                    why.append(f"{name} turns: {m} dropped {100 * d:.1f}% (the drafter lost context)")
                elif d > maybe:
                    soft.append(f"{name} turns: {m} dropped {100 * d:.1f}%: repeat with a second baseline run")
            elif d > 0.07:
                soft.append(f"{name} turns: {m} dropped {100 * d:.1f}% (decode is not what {b} changes: noise, repeat)")
        if numerics and name == "prose" and min(x["n"], y["n"]) < MIN_PROSE:
            soft.append(f"only {min(x['n'], y['n'])} prose turns per arm (need {MIN_PROSE}): acceptance is not judged")

    if A["skipped"] or B["skipped"]:
        print(f"skipped for time: {a} {A['skipped']} {b} {B['skipped']}")
    if why:
        return print("VERDICT: NO-GO (" + "; ".join(why) + ")")
    note = " [diagnostic arm: exactness, errors and cache behaviour only; time is not judged]" if diagnostic else ""
    print("VERDICT: " + ("INCONCLUSIVE (" + "; ".join(soft) + ")" if soft else "GO") + note)


if __name__ == "__main__":
    if len(sys.argv) in (4, 5) and sys.argv[1] == "compare":
        compare(*sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == "report":
        report(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] not in ("compare", "report"):
        arm(sys.argv[1])
    else:
        sys.exit(__doc__)
