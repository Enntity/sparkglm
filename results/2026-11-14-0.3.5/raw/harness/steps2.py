#!/usr/bin/env python3
"""steps2.py LOG: summarize ATLAS_MTP_STEP_TRACE ticks (plus owner/serial timing lines) of one benchmark case."""
import re, sys, collections, statistics as st
ticks = []; own = collections.defaultdict(list); ser = collections.defaultdict(list); prop = collections.defaultdict(list)
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"MTP STEP t=(\d+) dur=(\d+) kind=(\w+) n=(\d+) rows=(\d+) fwd=(\d+) nd=(\d+) deep=\d dl=(\d+) s=(\S+)(?: pf=(\d+))?", l)
    if m:
        seqs = [x.split(":") for x in m.group(9).split(",")]
        ticks.append(dict(dur=int(m.group(2)) / 1000, kind=m.group(3), n=int(m.group(4)), rows=int(m.group(5)), fwd=int(m.group(6)),
                          paths="".join(sorted(s[2][0] for s in seqs)), drafts=[int(s[2][1:]) for s in seqs], emitted=sum(int(s[3]) for s in seqs), pf=int(m.group(10) or 0) / 1000))
        continue
    m = re.search(r"GLM OWNER STEP_TIMING: owners=(\d+) rows=(\d+) sync=[\d.]+ms verify=([\d.]+)ms total=([\d.]+)ms", l)
    if m: own[(int(m.group(1)), int(m.group(2)))].append((float(m.group(3)), float(m.group(4)))); continue
    m = re.search(r"DFLASH STEP_TIMING: verify=([\d.]+)ms propose=([\d.]+)ms \(K=(\d+), accepted=(\d+)\)", l)
    if m: ser[int(m.group(3))].append((float(m.group(1)), float(m.group(2)))); continue
    m = re.search(r"GLM OWNER PROPOSE: owners=(\d+) ([\d.]+)ms", l)
    if m: prop[int(m.group(1))].append(float(m.group(2)))
ticks = [t for t in ticks if t["pf"] < 1 and t["kind"] == "mtp"]
byn = collections.defaultdict(list)
for t in ticks: byn[t["n"]].append(t)
for n, v in sorted(byn.items()):
    if len(v) < 20: continue
    dur = sum(t["dur"] for t in v); tok = sum(t["emitted"] for t in v)
    paths = collections.Counter(t["paths"] for t in v).most_common(4); fwd = collections.Counter(t["fwd"] for t in v).most_common(3)
    print(f"n={n}: ticks={len(v)} median_tick={st.median(t['dur'] for t in v):.0f}ms tok/tick={tok / len(v):.2f} -> {1000 * tok / dur:.1f} tok/s aggregate | paths {paths} | forwards/tick {fwd}")
for k, v in sorted(own.items()):
    if len(v) >= 5: print(f"  owner batch owners={k[0]} rows={k[1]}: n={len(v)} verify={st.median(x[0] for x in v):.1f}ms total={st.median(x[1] for x in v):.1f}ms")
for k, v in sorted(ser.items()):
    if len(v) >= 5: print(f"  single-sequence verify K={k}: n={len(v)} verify={st.median(x[0] for x in v):.1f}ms propose={st.median(x[1] for x in v):.1f}ms")
for k, v in sorted(prop.items()): print(f"  owner propose owners={k}: n={len(v)} {st.median(v):.1f}ms")
