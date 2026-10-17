# Reproduce our numbers

Run the drivers in [`bench/`](../bench/) on rank 0 against the running engine.
Run one workload at a time, with nothing else using the pair. They need only
Python 3; the comparison videos also need Pillow and ffmpeg.

The current release's numbers and raw receipts are in
[`results/2026-10-17-fp4local/`](../results/2026-10-17-fp4local/RESULT.md); earlier
releases' are in the other folders under [`results/`](../results/).

## Default profile (4 × 1M)

```sh
cd bench
# Matrix, three passes. Compare the sum of the five summary walls to 136.3 s.
for p in 1 2 3; do MATRIX_SALT=mb-vx2-7c31 python3 run_matrix.py out-matrix-$p matrix; done
# Staggered C4 field guide: a discarded warmup plus three runs. Use a fresh salt.
MATRIX_SALT=fg-$(date +%s) python3 run_matrix.py out-field field
# Quality probe: four-digit arithmetic and two-hop needles over ~24K tokens.
python3 quality_probe.py mine 4
```

`run_matrix.py` prints one JSON line per case. The number we sum is
`summary.wall_s`.

## 8 × 128K profile

Set `PROFILE=8x128k` in `.env` and rerun `./start.sh`. We measured the
8-stream aggregate and RigMark on this profile. Run
[RigMark](https://github.com/othexmr/rigmark) with the cell set in
[`bench/RIGMARK.md`](../bench/RIGMARK.md).

## FP4-Local release (2026-10-17)

ToolEvalBench ([tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
at `471c252`, default profile):

```sh
tool-eval-bench run --base-url http://127.0.0.1:8893 --model glm-5.3-flash-atlas \
  --seed 42 --trials 3 --parallel 4
```

Kindling's decode benchmark and cold prefill, from a checkout of
[kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10)
(its published figures we compare against are at `0d1f4620`; its code is not included here). `bench.py` ran on
`PROFILE=8x128k`, `prefill.py` on `4x512k`:

```sh
GATE_URL=http://127.0.0.1:8893 GATE_MODEL=glm-5.3-flash-atlas \
  python3 gate/bench/bench.py run --label mine --out kb-mine.json
python3 gate/prefill.py --base http://127.0.0.1:8893/v1 --model glm-5.3-flash-atlas \
  --sizes 32k,128k --skip-thinking --out-tokens 32 --seed 11
```

The harness that ran every arm is in
[`results/2026-10-17-fp4local/raw/harness/`](../results/2026-10-17-fp4local/raw/harness/).

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
