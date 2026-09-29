#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""The SparkGLM matrix (G3: C1/C2 at 16K/32K, C4 at 16K) and the staggered C4
field guide, straight at the engine (127.0.0.1:8893 by default).

Usage: run_matrix.py OUT_DIR [matrix,field] [CASE,...]
Same drivers, salt, 400-token output budget, one-second stagger and exact
prompt counts as the published runs. Use MATRIX_SALT=mb-vx2-7c31 for the
matrix; give the field guide a fresh salt per run (e.g. fg-$(date +%s)).
"""
import os
import json, pathlib, subprocess, sys, time
B = pathlib.Path(__file__).resolve().parents[4] / "benchmarks"
URL = os.environ.get("MATRIX_BASE_URL", "http://127.0.0.1:8893")
MODEL = os.environ.get("MATRIX_MODEL", "glm-5.3-flash-atlas")
SALT = os.environ.get("MATRIX_SALT", "mb-afd137970d6a")
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
suites = sys.argv[2].split(",") if len(sys.argv) > 2 else ["matrix", "field"]
cases = []
if "matrix" in suites:
    cases += [("g3", n, c, t) for n, c, t in
              [("c1-16k", 1, 16384), ("c1-32k", 1, 32768), ("c2-16k", 2, 16384),
               ("c2-32k", 2, 32768), ("c4-16k", 4, 16384)]]
if "field" in suites:
    cases += [("c4", n, 4, 16384) for n in
              ["discarded-warmup", "field-guide-r1", "field-guide-r2", "field-guide-r3"]]
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
for kind, name, c, t in cases:
    if only and name not in only:
        continue
    f = out / f"{name}.json"
    if kind == "g3":
        argv = ["python3", str(B / "staggered_openai.py"), "--base-url", URL, "--model", MODEL,
                "--concurrency", str(c), "--prompt-tokens", str(t), "--exact-prompt-tokens",
                "--prompt-salt", f"{SALT}-{name}", "--output-tokens", "400",
                "--min-output-tokens", "400", "--stagger-ms", "1000", "--timeout-s", "900"]
        t0 = time.time()
        with f.open("w") as fh:
            rc = subprocess.run(argv, stdout=fh, stderr=subprocess.PIPE, text=True).returncode
        s = json.loads(f.read_text())["summary"] if rc == 0 else {}
        print(json.dumps({"case": name, "rc": rc, "wall_s": round(time.time() - t0, 2),
                          "summary": {k: s.get(k) for k in ("successful", "wall_s", "aggregate_decode_tok_s",
                                                            "median_ttft_s", "max_ttft_s") if k in s}}), flush=True)
    else:
        argv = ["python3", str(B / "four_stream_video.py"), "capture", "--base-url", URL,
                "--model", MODEL, "--output", str(f), "--streams", "4", "--max-tokens", "400",
                "--prompt-tokens", "16384", "--stagger-ms", "1000", "--prompt-style", "field-guide",
                "--cache-salt", f"{SALT}-{name}", "--timeout-s", "900"]
        t0 = time.time()
        rc = subprocess.run(argv, capture_output=True, text=True).returncode
        d = json.loads(f.read_text()) if rc == 0 and f.exists() else {}
        print(json.dumps({"case": name, "rc": rc, "wall_s": round(time.time() - t0, 2),
                          "total_completion_tokens": d.get("total_completion_tokens"),
                          "reported_wall_s": d.get("wall_s") or d.get("total_wall_s")}), flush=True)
