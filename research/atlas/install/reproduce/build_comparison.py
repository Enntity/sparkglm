#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Select each recipe's median field-guide run and write a render_comparison.py input.

Usage: build_comparison.py OUT.json "TITLE|SUBTITLE|run1.json,run2.json,run3.json" [...]
Each arm lists that recipe's retained runs (four_stream_video.py captures). Every
request must have produced exactly 400 completion tokens.
"""
import json, pathlib, statistics, sys

arms = []
for spec in sys.argv[2:]:
    title, subtitle, files = spec.split("|")
    runs = [json.loads(pathlib.Path(p).read_text()) for p in files.split(",")]
    for r in runs:
        assert all(q["completion_tokens"] == 400 and q["error"] is None for q in r["requests"]), title
    walls = sorted(r["wall_s"] for r in runs)
    median = statistics.median(walls)
    run = min(runs, key=lambda r: abs(r["wall_s"] - median))
    toks = [q["prompt_tokens"] for q in run["requests"]]
    arms.append({"title": title, "subtitle": subtitle,
                 "detail": f"Median run · actual input {min(toks):,}–{max(toks):,} tokens/request · whole workload {run['wall_s']:.2f}s",
                 "capture": run})
    print(f"{title}: runs {[round(w, 2) for w in walls]} -> {run['wall_s']:.2f}s")
pathlib.Path(sys.argv[1]).write_text(json.dumps(arms))
