# Reproduce our numbers

Run the drivers in [`bench/`](../bench/) on rank 0 against the running engine.
Run one workload at a time, with nothing else using the pair. They need only
Python 3; the comparison videos also need Pillow and ffmpeg.

The published numbers and their raw receipts are in
[`results/2026-09-29-atlas-merged/`](../results/2026-09-29-atlas-merged/RESULT.md).

## Default profile (4 × 512K)

```sh
cd bench
# Matrix, three passes. Compare the sum of the five summary walls to 154.8 s.
for p in 1 2 3; do MATRIX_SALT=mb-vx2-7c31 python3 run_matrix.py out-matrix-$p matrix; done
# Staggered C4 field guide: a discarded warmup plus three runs. Use a fresh salt.
MATRIX_SALT=fg-$(date +%s) python3 run_matrix.py out-field field
# Quality probe: four-digit arithmetic and two-hop needles over ~24K tokens.
python3 quality_probe.py mine 4
# mmastrac's prefill/decode benchmark, then 1/2/4/8 aggregate streams.
python3 mmastrac_bench.py && python3 mmastrac_streams.py
```

`run_matrix.py` prints one JSON line per case. The number we sum is
`summary.wall_s`.

## 8 × 128K profile

Set `PROFILE=8x128k` in `.env` and rerun `./start.sh`. We measured the
8-stream aggregate and RigMark on this profile. Run
[RigMark](https://github.com/othexmr/rigmark) with the cell set in
[`bench/RIGMARK.md`](../bench/RIGMARK.md).

## Comparison video

This compares our field guide against another recipe's captures:

```sh
python3 build_comparison.py cmp.json \
  "Atlas|this recipe|out-field/field-guide-r1.json,out-field/field-guide-r2.json,out-field/field-guide-r3.json" \
  "Other recipe|its settings|a1.json,a2.json,a3.json"
python3 render_comparison.py cmp.json atlas-c4.mp4 "Atlas vs other · staggered C4"
```

The vLLM SparkGLM and Mia EXL3 captures we compared against are in
`results/2026-09-29-atlas-merged/raw/baselines/`. `four_stream_video.py`
records four live streams as a grid video (`python3 four_stream_video.py --help`).
