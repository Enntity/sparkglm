#!/usr/bin/env python3
"""vg-probe.py ARM | compare — ATLAS_GLM_VERIFY_GRAPH A/B probe (runs on spark1).

ARM runs five cells against the atlas-dev pair and writes ~/vg-ARM.json:
  short   16 sequential x 64 forced tokens, FIRST after start (carries every
          capture step: the per-request capture cost shows here)
  greedy  code + facts prompts, 2 x 256 tokens each (byte-equality)
  c1      5 x 384 forced tokens, prose (decode tok/s; texts are the prose
          byte-equality check; replay-only once `short` has warmed the slot)
  c2      2 concurrent x 384 forced tokens
  c4      4 concurrent x 384 forced tokens
For each cell it reads the rank-0 atlas-dev log lines the cell produced:
piecewise captures (count, ms, slots, peak cached), refusals, and the
per-request `Done:` tok_step (tokens per verify step = mean accepted + 1).
compare: verdicts from ~/vg-off.json and ~/vg-on.json.
"""
import json, os, re, statistics, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8893/v1/chat/completions"
MODEL = "glm-5.3-flash-atlas"
PROSE = "<prompt omitted: quoted from mmastrac/glm-5.3-flash-4x-gx10 @ 4e63b64, which has no license>"
GREEDY = {
    "code": "<prompt omitted: quoted from mmastrac/glm-5.3-flash-4x-gx10 @ 4e63b64, which has no license>",
    "facts": "List the planets of the solar system in order from the sun, one sentence each.",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CAPTURE = re.compile(r"piecewise verify graph \(slot, rows, layer\) = \((\d+), (\d+), (\d+)\): "
                     r"(\d+) graphs in ([0-9.]+) ms \((\d+) graphs cached\)")
DONE = re.compile(r"Done: (\d+) tokens .*? tok_step=([0-9.]+)")


def gen(prompt, max_tokens, forced=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    if forced:
        body["min_tokens"] = max_tokens
    t0, text, first, last, n = time.time(), "", None, None, 0
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            n = (ev.get("usage") or {}).get("completion_tokens", n)
            for ch in ev.get("choices") or []:
                c = (ch.get("delta") or {}).get("content")
                if c:
                    now = time.time() - t0
                    text, first, last = text + c, first if first is not None else now, now
    rate = (n - 1) / (last - first) if last and last > first else 0.0
    return {"text": text, "n": n, "rate": round(rate, 2), "wall": round(time.time() - t0, 3)}


def log_lines():
    out = subprocess.run(["docker", "logs", "atlas-sparkglm-rank0"], capture_output=True, text=True, errors="replace")
    return [ANSI.sub("", l) for l in (out.stdout + out.stderr).splitlines()]


def cell(fn):
    """Run fn(); return (its result, stats of the rank-0 log lines it produced)."""
    n0 = len(log_lines())
    res = fn()
    time.sleep(1.5)  # the scheduler logs `Done:` just after the last chunk
    lines = log_lines()[n0:]
    caps = [m.groups() for m in map(CAPTURE.search, lines) if m]
    per_req, acc = [], 0.0  # capture ms between consecutive `Done:` lines
    for l in lines:
        m = CAPTURE.search(l)
        if m:
            acc += float(m.group(5))
        elif DONE.search(l):
            per_req.append(round(acc, 1))
            acc = 0.0
    return res, {
        "captures": len(caps),
        "capture_ms": round(sum(float(c[4]) for c in caps), 1),
        "capture_ms_per_request": per_req,
        "capture_slots": sorted({int(c[0]) for c in caps}),
        "peak_cached": max((int(c[5]) for c in caps), default=0),
        "refused": sum("piecewise verify graph refused" in l for l in lines),
        "tok_step": [float(m.group(2)) for m in map(DONE.search, lines) if m],
    }


def concurrent(n, tokens):
    with ThreadPoolExecutor(n) as ex:
        t0 = time.time()
        res = list(ex.map(lambda i: gen(f"[{i}] " + PROSE, tokens, forced=True), range(n)))
    return {"per_stream": [r["rate"] for r in res],
            "aggregate": round(sum(r["n"] for r in res) / (time.time() - t0), 2)}


def arm(name):
    out = {}
    res, out["short_log"] = cell(lambda: [gen(f"[s{i}] " + PROSE, 64, forced=True) for i in range(16)])
    out["short"] = {"wall_s": round(sum(r["wall"] for r in res), 2), "walls": [r["wall"] for r in res]}
    res, out["greedy_log"] = cell(lambda: {k: [gen(p, 256)["text"] for _ in range(2)] for k, p in GREEDY.items()})
    out["greedy"] = res
    res, out["c1_log"] = cell(lambda: [gen(PROSE, 384, forced=True) for _ in range(5)])
    out["c1"] = [r["rate"] for r in res]
    out["greedy"]["prose"] = [r["text"] for r in res]
    out["c2"], out["c2_log"] = cell(lambda: concurrent(2, 384))
    out["c4"], out["c4_log"] = cell(lambda: concurrent(4, 384))
    json.dump(out, open(os.path.expanduser(f"~/vg-{name}.json"), "w"), indent=1)
    print(json.dumps({"arm": name, "c1": out["c1"], "c1_tok_step": out["c1_log"]["tok_step"],
                      "short_wall_s": out["short"]["wall_s"], "c2": out["c2"], "c4": out["c4"],
                      **{f"{c}_captures": out[f"{c}_log"]["captures"] for c in ("short", "greedy", "c1", "c2", "c4")}}))


def spread(xs):
    return f"min {min(xs):.2f} / median {statistics.median(xs):.2f} / max {max(xs):.2f}"


def compare():
    off, on = (json.load(open(os.path.expanduser(f"~/vg-{a}.json"))) for a in ("off", "on"))
    for key in ("code", "facts", "prose"):
        a, b = off["greedy"][key], on["greedy"][key]
        ref = max(set(a), key=a.count)  # the off arm's modal text
        eq = sum(t == ref for t in b)
        verdict = "EQUAL" if eq == len(b) else (
            "INCONCLUSIVE(off not self-equal)" if len(set(a)) > 1 else "MISMATCH")
        print(f"greedy {key}: {verdict} on runs equal to off: {eq}/{len(b)} (off distinct texts: {len(set(a))})")
    lo, hi = min(on["c1"]), max(on["c1"])
    verdict = ("GO" if lo >= max(off["c1"]) else
               "REGRESSION" if hi <= min(off["c1"]) else "INCONCLUSIVE")
    print(f"C1 tok/s off {spread(off['c1'])} | on {spread(on['c1'])} | median "
          f"{(statistics.median(on['c1']) / statistics.median(off['c1']) - 1) * 100:+.1f}% -> {verdict}")
    # Acceptance: runs whose text equals the off arm's modal prose must accept
    # identically; a graph-baked hidden-capture fault shows up here first.
    ref = max(set(off["greedy"]["prose"]), key=off["greedy"]["prose"].count)
    pair = lambda d: {s for t, s in zip(d["greedy"]["prose"], d["c1_log"]["tok_step"]) if t == ref}
    ts_off, ts_on = pair(off), pair(on)
    acc = ("ACCEPT-EQUAL" if ts_on and ts_on <= ts_off else
           "ACCEPT-DRIFT" if ts_on else "ACCEPT-UNPAIRED(compare means)")
    mean = lambda d: statistics.mean(d["c1_log"]["tok_step"]) if d["c1_log"]["tok_step"] else 0.0
    print(f"C1 tok_step off {sorted(ts_off)} on {sorted(ts_on)} (byte-equal runs); "
          f"means {mean(off):.3f} -> {mean(on):.3f} -> {acc}")
    print(f"short 16x64 wall {off['short']['wall_s']} -> {on['short']['wall_s']} s "
          f"({(on['short']['wall_s'] / off['short']['wall_s'] - 1) * 100:+.1f}%); on capture ms per request "
          f"{on['short_log']['capture_ms_per_request']}")
    for c in ("c2", "c4"):
        print(f"{c.upper()} aggregate {off[c]['aggregate']} -> {on[c]['aggregate']} tok/s "
              f"({(on[c]['aggregate'] / off[c]['aggregate'] - 1) * 100:+.1f}%)")
    for c in ("short", "greedy", "c1", "c2", "c4"):
        s = on[f"{c}_log"]
        print(f"on {c}: captures {s['captures']} ({s['capture_ms']} ms), slots {s['capture_slots']}, "
              f"peak cached {s['peak_cached']}, refused {s['refused']}")


if __name__ == "__main__":
    compare() if sys.argv[1] == "compare" else arm(sys.argv[1])
