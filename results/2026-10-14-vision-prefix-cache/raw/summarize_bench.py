#!/usr/bin/env python3
"""summarize_bench.py DIR: the release-suite table rows from one benchmark directory holding rig-decode.json,
rig-conc.json, field-16k/, field-32k/ and matrix/ (the layout of results/*/raw)."""
import glob, json, os, statistics, sys

d = sys.argv[1]
J = lambda p: json.load(open(os.path.join(d, p)))
dec = J("rig-decode.json"); conc = J("rig-conc.json")
med = lambda x: x["median"]
rows = {
    "rigmark decode code/prose/structured tok/s": [med(dec["decode"][w]["decode_tokens_per_second"]) for w in ("code", "prose", "structured")],
    "rigmark cold prefill 8K/32K/64K TTFT s": [med(dec["prefill"][k]["cold"]["ttft_seconds"]) for k in ("8192", "32768", "65536")],
    "rigmark staggered prefill-first newcomer TTFT at 2/4 s": [med(conc["staggered"][k]["prefill_first"]["newcomer_median_ttft_seconds"]) for k in ("2", "4")],
    "rigmark short code 1/2/3/4 streams aggregate tok/s": [med(conc["concurrency"][k]["aggregate_end_to_end_tokens_per_second"]) for k in ("1", "2", "3", "4")],
}
for n in ("16k", "32k"):
    walls = [json.load(open(f))["wall_s"] for f in sorted(glob.glob(os.path.join(d, f"field-{n}", "field-guide-r*.json")))]
    rows[f"field guide {n} median wall s (n={len(walls)})"] = [round(statistics.median(walls), 1)] if walls else []
mats = sorted(glob.glob(os.path.join(d, "matrix", "*.json")))
rows["matrix sum of walls s"] = [round(sum(json.load(open(f))["summary"]["wall_s"] for f in mats), 1)]
for k, v in rows.items():
    print(f"{k}: {' / '.join(str(round(x, 2)) for x in v)}")
