# 2026-09-29: Atlas SparkGLM from a fresh clone

> **This configuration had no prefix caching.** Every request, including each
> turn of a multi-turn conversation, prefilled its whole prompt from scratch;
> RigMark's immediate-replay cells below show it. It is superseded by
> [2026-09-29-prefix-caching](../2026-09-29-prefix-caching/RESULT.md).

## Hypothesis

The merged Atlas engine — `Enntity/atlas` `sparkglm/atlas-20260928`: Atlas-Inf
`main` plus our GLM-5.3 Flash layer and one SparkGLM-only bridge commit — can
be installed by anyone from pinned public sources with this repository's
recipe, and the installed pair serves the frozen matrix,
the staggered C4 field guide and RigMark at least as fast as our earlier
Atlas fork and faster than the vLLM SparkGLM configuration.

## Frozen configuration

- **Recipe:** this repository at `a2ca292e25228ef3021a4b80a17298c299c10017`
  (branch `atlas/installable-20260929`), followed as written from a fresh clone
  on both nodes: `build.sh`, overlay conversion and verification, then
  `start-node.sh` (worker first). That commit predates the repository's
  reorganization; the same install files now live in `install/`, and the
  benchmark drivers in `bench/`.
- **Engine:** `Enntity/atlas` @ `c9723935592f3d12215a77b0dfdfcadf641a567f`
  (tree `4cd94911f4911085969ea508d0668cd5446dd06c`), kernel target
  `glm-5.3-flash`. Image `lloom/atlas-sparkglm:a2ca292e…` built on each node:
  rank 0 `sha256:19a390e0…`, rank 1 `sha256:2990a301…` (`raw/cleanroom/rank*/`).
  BuildKit caches were warm from earlier builds (build 310/315 s).
- **Model:** `nvidia/GLM-5.3-Flash-NVFP4` @ `423acf37`; drafter
  `incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd8`. The overlay was converted
  freshly (864 matrices, 9,962 verified payloads, 65–69 s per node).
- **Profile:** `install/profiles/4x512k.json`: TP2/EP2, 4 × 512K,
  FP8-latent KV (866,208 tokens), KDA record rollback, DFlash2 γ=8 with
  adaptive width, GPU utilization 0.88. RigMark and the 8-stream figure use
  `install/profiles/8x128k.json` (8 × 128K).
- **Hardware:** two DGX Spark GB10 (128 GB), one 200G ConnectX-7 cable, RDMA
  all-reduce over both PCIe halves. Requests went straight to the engine on
  127.0.0.1:8893, one workload at a time.

## Results

**Frozen matrix** (`raw/cleanroom/out-matrix-*`; summary wall, seconds; prompts
calibrated to exactly 16,384 / 32,768 tokens; 400 output tokens each):

| Case | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---|
| C1-16K | 15.06 | 14.36 | 14.15 |
| C1-32K | 21.41 | 21.01 | 21.33 |
| C2-16K | 26.61 | 28.69 | 29.62 |
| C2-32K | 42.94 | 44.06 | 42.15 |
| C4-16K | 46.39 | 51.10 | 45.39 |
| **Total** | **152.40** | **159.22** | **152.65** |

The mean total is **154.8 s**. References on the same pair: the fork Atlas
build (3a6de55f) 155.4 s, vLLM SparkGLM 206.6 s. Pass 1 includes the first
use of a fresh CUDA kernel cache (C1-16K decode 46.5 tok/s against 52–54 in
passes 2–3); no separate warmup was discarded for the matrix.

**Staggered C4 field guide** (`raw/cleanroom/out-field`; four ~16K requests
arriving at 0/1/2/3 s, 400 tokens each; one discarded warmup):

| Recipe | Runs (s) | Median |
|---|---|---|
| **Atlas, this recipe** | 57.54 / 56.70 / 58.93 | **57.54** |
| vLLM SparkGLM NVFP4, adaptive K2/4/5 (2026-09-20) | 68.64 / 71.22 / 70.51 | 70.51 |
| Mia EXL3 stock TP2, live `0f49cfd` (2026-09-24) | 107.09 / 115.24 / 113.22 | 113.22 |

The two reference arms (`raw/baselines/`) were recorded earlier on the same
pair through the LLooM gateway; they were not alternated with this run.
Their prompts rendered 1–7 tokens differently because of chat templates.

**mmastrac benchmark** (single stream, 512 tokens, median of 3):
decode structured / code / prose **84.1 / 60.5 / 32.2 tok/s**; prefill
35,237 tokens in 15.0 s (**2,349 tok/s**) and 139,273 tokens in 69.0 s
(**2,018 tok/s**). Aggregate streams on the 4 × 512K profile: 82.1 / 84.8 /
83.5 / 81.4 tok/s at 1 / 2 / 4 / 8 (8 streams queue behind four slots);
8 × 128K profile: 85.8 / 87.0 / 76.9 / **122.5** tok/s.

**RigMark** (8 × 128K, `reasoning_effort: low`, RiNGSiDE TP2 cell set, two
runs per cell): both receipts pass **6/6** basic output gates (`raw/cleanroom/rig-*.json`,
summary in `rigmark-summary.txt`).

| Cell | This recipe | RiNGSiDE TP2 (published) |
|---|---|---|
| Decode code / prose / structured (tok/s) | 52.5 / 26.5 / 79.2 | 56.5 / 33.0 / 83.4 |
| Short-code aggregate C1 / C4 / C6 / C8 (tok/s) | 36.9 / 62.1 / 76.7 / 88.0 | 44.0 / – / 97.3 / – |
| Short-prose aggregate C1 / C4 / C6 / C8 (tok/s) | 25.4 / 43.0 / 55.5 / 64.7 | – |
| Cold prefill 8K / 32K / 64K (s) | 3.27 / 13.25 / 28.33 | 3.56 / 12.94 / 25.66 |
| Immediate replay of the same prompt, 8K / 32K / 64K (s) | 3.27 / 13.24 / 28.31 (no prefix caching: same as cold) | ~2.3–2.7× faster than its cold prefill |
| Prefill-first newcomer TTFT L2 / L4 / L6 / L8 (s) | 2.79 / 3.30 / 3.80 / 4.31 | 4.70 / 5.15 / 4.90 / – |
| Decode-first newcomer TTFT L2 / L4 / L6 / L8 (s) | 13.59 / 13.86 / 14.05 / 13.84 | 14.38 / 14.81 / 15.42 / – |

The RiNGSiDE column is that project's published TP2 row (vLLM 0.29), not a
run on our pair.

## Correctness

- **Quality probe** (`raw/cleanroom/qprobe-cleanroom.jsonl`): 40/40 four-digit
  multi-step arithmetic, 11/12 two-hop needle retrievals over ~24K tokens
  (the fork build scores 40/40 and 10–12/12).
- **Smoke test:** `17*23` answered `391`.
- **Build-time gates:** the image build ran the engine's DFlash verify-policy
  (2), thinking-budget (7), admission (11), lifecycle (13), tool-stop (7) and
  vision-guard (2) tests.
- **Focused behaviour on the same engine commit, dev launch:**
  `tool_choice=required` returns a well-formed call; presence and frequency
  penalties apply; a JSON answer under a 128-token thinking budget arrives as
  content.

## Limitations

- **Reference arms:** the matrix and field-guide runs are
  complete-configuration measurements with three repetitions, but the vLLM
  and Mia arms are earlier recordings on the same pair, not runs alternated
  with this one.
- **Scope:** one pair of Sparks. Other cables, firmware or driver versions may
  differ.
- **Not measured with this recipe image:** long-context behaviour. On the same
  engine commit before the thinking-budget fix, four concurrent 190K requests
  and one 500K request completed with at least 8.3 GB of host memory free.
  Endurance, cancellation under load and multimodal work at full context were
  not measured.
- **Build time:** a cold build was not timed; the recorded builds reused
  BuildKit caches.
- **Remaining gap to the vLLM TP2 recipe by RiNGSiDE on RigMark:**
  single-stream decode, low-concurrency short code and long cold prefill.

## Verify the receipts

```sh
cd results/2026-09-29-atlas-merged && sha256sum -c SHA256SUMS
```
