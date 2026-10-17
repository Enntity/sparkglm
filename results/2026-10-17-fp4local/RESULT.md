# 2026-10-17: FP4-Local (sparkglm/atlas-20261017-fp4local)

The default checkpoint is now NVIDIA's
[`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) at `76348ddb` (165.3 GiB),
replacing [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) at `423acf37` (190.4 GiB).
The engine is the release tag `sparkglm/atlas-20261017-fp4local` at `b3b016ab` (tree `14316380`): the shared-engine
release (`16315004`) plus the FP4-Local series. Its tree is identical to `dc6084a5`, the commit the gates below ran.
FP4-Local ships its MTP layer quantized, so the 16 GB MTP overlay `convert.sh` wrote for the NVFP4 checkpoint is
gone. The NVFP4 checkpoint still works on the same engine, through its overlay, and scores production's prompt
logprobs bit for bit. The recipe's image is `ghcr.io/enntity/atlas-sparkglm:c8ced64316bf`
([Release image](#release-image)): the dev gates' prompt logprobs, tools 21/21, ToolEvalBench 91.

## The checkpoint

Both are NVIDIA ModelOpt quantizations of GLM-5.3-Flash. FP4-Local is quantization-aware distilled against the BF16
model and keeps less in BF16:

| | NVFP4 (`423acf37`) | FP4-Local (`76348ddb`) |
|---|---|---|
| Routed experts | NVFP4, E4M3 block scales | NVFP4 (QAD), the block scales stored losslessly compressed (`csf_32x2_v1`) |
| KDA `q/k/v/o`, dense MLP (layers 0–2), `lm_head` | BF16 (dense MLP NVFP4) | MXFP8 (E4M3 + one power-of-two scale per 1x32) |
| MLA `q_a/q_b/kv_a/o`, shared experts | BF16 | FP8_PB_WO (E4M3 + one F32 scale per 128x128) |
| MTP layer | BF16; Atlas needed an overlay quantizing 864 matrices | MXFP8, routed experts W4A16 NVFP4 |
| Vision tower | BF16 | MXFP8 attention, W4A16 NVFP4 MLP |
| Size | 190.4 GiB | 165.3 GiB, including one shard in a subdirectory (`tensors/native-QB.safetensors`) |
| License | MIT | NVIDIA Open Model Agreement (base model MIT) |

The DFlash2 drafter is unchanged (`incoai/GLM-5.3-Flash-DFlash2` at `7d74cdd8`).

## What changed

**Engine** (112 commits on `16315004`; those marked `shared:` are model-independent and sit in the `sparkglm` layer
until the shared layer moves):

- **Lossless scale decoding.** The routed experts' `csf_32x2_v1` block scales are decoded at upload, and under expert
  TP per rank slice from the deferred components; every decoded scale is checked against its weight's shape.
- **FP4-Local through production's kernels.** Each MXFP8 or FP8_PB_WO projection is fed to the arm production runs on
  the NVFP4 checkpoint, from an exact (MXFP8) or nearest (FP8 block, within one BF16 half-ulp) BF16 copy; the
  quantized sources are released once a layer is built (6.83 GiB per rank). The MTP experts and the ViT load from
  their W4A16 NVFP4 weights.
- **Four exact prefill options, now in every profile:** KDA `beta | f_a | g_a` as one packed CUTLASS GEMM
  (`ATLAS_GLM_KDA_SIDE_PROJ_LT`), KDA convolution over the q/k/v planes without the packed copy
  (`ATLAS_GLM_KDA_CONV_PLANES`), the mHC pre-site finalizer writing the block's input norm
  (`ATLAS_GLM_HC_PREFILL_FUSE2`) and sequence-parallel seams that overlap their exchanges with compute
  (`ATLAS_GLM_SP_GATHER_OVERLAP`). The prompt-logprob hash is the same with each on.
- **Tools with `response_format`.** Under `tool_choice: auto`, a request with both tools and a `response_format`
  schema now compiles one tool-call-or-JSON union grammar: the model may call a tool or answer in the schema, and the
  union picks its branch by probability mass. Before, four ToolEvalBench scenarios that send both scored partial.
- **Default-off options**, described in [docs/ENGINE.md](../../docs/ENGINE.md): every group in the checkpoint's own
  format (`ATLAS_GLM_FP4L_NATIVE`), the drafter's prompt window on warm turns (`ATLAS_DFLASH_WARM_CTX`), packed
  multi-prompt prefill (`ATLAS_GLM_PACKED_PREFILL`), `ATLAS_TAIL_SPLIT_MIN_TOKENS` and a per-request TTFT trace
  (`ATLAS_TTFT_TRACE`).

**Recipe:** `start-node.sh --checkpoint` names the checkpoint directory (default FP4-Local, served as published).
`.env` can set `CHECKPOINT_REPO` and `CHECKPOINT_REVISION` (both or neither); for `nvidia/GLM-5.3-Flash-NVFP4` alone
`./start.sh` still converts the overlay and serves it. `serve.py` refuses to start unless the MTP layer is quantized in
the checkpoint's config or converted by the overlay.

## Setup

ennspark01/02, one 200G cable, GPU clocks uncapped, `GPU_MEMORY_UTILIZATION=0.91`, display carveout, KV shard and a
48 GB disk prefix cache, each arm from a fresh start with an empty cache directory (`raw/harness/up.sh`). The engine
ran as dev binaries of the FP4-Local series on the shared-engine image `f0f8a55c3da2`, through a copy of
`start-node.sh` that mounts the binary and the checkpoint directory, so each log's `atlas-recipe-start` line names that
image's source (`16315004`). The binary for each arm is named below. The [release image](#release-image) built from
this recipe was measured afterwards with the same settings.

## Gates

The release candidate `dc6084a5` on `4x1m` with the four switches, the profile this recipe ships
(`raw/gates/4x1m.json`, harness `raw/harness/gates2.sh`):

| Gate | Result |
|---|---|
| Quality probe: multi-step arithmetic / two-hop 24K needles | 39/40 · 12/12, as published for RC4 (`qprobe-rc2.jsonl`) |
| `tool_probe.py`, 7 cases x 3 (single, typed, parallel, follow-up, no tool, `tool_choice: none`, streamed) | 21/21 (`tools-rc2.json`) |
| Greedy text at concurrency 1 and 4 | 8/8 identical (`greedy-rc2-c1.json`, `-c4.json`) |
| Greedy C4 on a shared ~4.4K-token prefix, 3,712–3,728 tokens restored from the cache per request | 8/8 identical to the same run on the g2 binary's start (`greedy-rc2-c4p.json`, `greedy-g2-c4p.json`) |
| Prompt logprobs of the 48,875-token text, two runs | `25bc828eea3f` both times (`lp-glm-rc2.json`) |
| Prefix-cache test | 8/8: a 35K replay in 0.38 s (13.37 s cold); turns 2–6 of a 45K conversation in 0.65–0.91 s (`pc-glm-rc2.json`) |
| Four concurrent ~204K-token sessions x 3 turns | 12/12 numbers retrieved, 10/12 strict, in 311.7 s (`fill-glm-rc2.json`) |
| Lowest rank-0 MemAvailable during the fill | 3.29 GB (`mem-fill-rc2.log`); 4.0–4.6 GB with the NVFP4 checkpoint on 2026-10-15 |
| Panics or errors | none (`serve-rc2.rank0.log` keeps the load and every warning) |

The fill's two strict misses are a probe artifact: its 24-token cap cut two correct six-digit answers after their
first three digits (`535` of 535903, `455` of 455196). The answer is the right number in all twelve.

**Prompt logprobs are reproducible and unchanged by every prefill switch:** `25bc828eea3f` with no switch, with three
and with all four, on every binary that ran it (`raw/prefill/*/lp-glm-*.json`, `raw/gates/lp-glm-g2-N.json`). Prefill
in 16K chunks gives another hash (`46cdaad4c42f`), since prefill is not yet chunk-invariant. On the same binary the
NVFP4 checkpoint through its overlay scores `246cd1331b41`, production's and RC4.2's hash (`lp-glm-g2-O.json`).

## KV pool

Same binary, same settings (`4x1m`, 0.91, carveout, shard, disk prefix cache), back-to-back starts
(`raw/gates/serve-g2-*.rank0.log`):

| | FP4-Local | NVFP4 + overlay |
|---|---:|---:|
| Rank 0 before the KV pool (free-memory measure) | 101.94 GB | 102.45 GB |
| KV pool | **166,044 blocks, 2.66M tokens** | 146,512 blocks, 2.34M tokens |

The pool comes from a free-memory measurement that moves between starts: FP4-Local's production-mode starts with the shipped switches in these
receipts gave 157,655–167,874 blocks at 0.91 (a start without the switches, `raw/prefill/p2-base`, gave 173,028) (the release candidate's: 157,950), and the release image's `4x1m` start
181,120 (2.90M tokens; its carveout took two buffers). Prefill in 16K chunks costs about
4.1 GB more before the KV pool and halves it (77,348 blocks, `raw/prefill/ov-on16k/serve.rank0.log`).

## ToolEvalBench

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) (SeraphimSerapis) at `471c252`, the 69
standard scenarios, seed 42, temperature 0, thinking on, 3 trials, `--parallel 4`, `4x1m` (`raw/toolevalbench/`):

| Engine | Score | Points |
|---|---:|---:|
| FP4-Local, before the tools + `response_format` fix (`teb-prod`) | 88.0 ± 0.0 | 122/138 |
| FP4-Local, with the fix (`teb-rfmt2`) | **91.0 ± 0.0** | 126/138 |
| FP4-Local in its own formats (`ATLAS_GLM_FP4L_NATIVE=1`, `teb-faithful`, one trial) | 91 | 125/138 |
| NVIDIA's model card, FP4-Local / BF16 (an earlier 63-scenario version of the suite) | 92.4 / 91.9 | |

The fix turned TC-65, TC-66, TC-67 and TC-69 from partial to pass in every trial. The remaining misses are the
model's choices (for example TC-51 sends the email before the calendar event it depends on returns), not
transport errors. TC-68 sends no `response_format`, so the grammar cannot help there. In both three-trial runs every
trial scored the same total, though four scenarios moved between trials.

## Speed

All on `8x128k` or `4x512k` at 0.91 as noted.

**[kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10)'s `gate/bench/bench.py`** (our
outputs only; their code is not included), 16 runs a workload (8 batches for mixed), thinking off, temperature 0,
`8x128k`, an earlier build of the series without the prefill switches (`raw/bench/kb-f1.json`). Their figures are
their TP=2 vLLM recipe on two GB10 boxes with the NVFP4 checkpoint, as published in `gate/bench/README.md` (mixed rows
remeasured in their README on 2026-10-16, in parentheses):

| tok/s (bootstrap 95% interval) | SparkGLM, FP4-Local | kindling TP=2, published |
|---|---:|---:|
| code | 72.8 (68.8–77.0) | 66.2 (61.5–71.6) |
| prose | 44.4 (43.3–45.7) | 42.6 (41.7–43.6) |
| structured | 99.5 (95.6–103.5) | 94.7 (89.1–100.4) |
| counting | 112.2 (110.4–114.1) | 97.2 (96.0–98.5) |
| mixed, 4 streams | 82.6 (78.0–87.4) | 82.6 (89) |
| mixed, 8 streams | 100.6 (96.8–104.6) | 109.6 (116) |

At eight concurrent streams their recipe is ahead.

**Cold prefill**, their `prefill.py` (random words, nothing cached), `4x512k`, one prompt per size per start
(`raw/prefill/`):

| tok/s, 32K / 128K | Prefill chunk | |
|---|---|---:|
| No switch (`p2-base`) | 8K | 2,581 / 2,674 |
| KDA side GEMM + conv planes + mHC finalizer (`p2-all`) | 8K | 2,645 / 2,733 |
| The same + SP overlap (`ov-on`; 32K only; `ov-base` without it: 2,622) | 8K | 2,690 / – |
| Three switches, 16K chunks (`p2-all16k`) | 16K | 2,798 / 2,916 |
| All four, 16K chunks (`ov-on16k`) | 16K | 2,886 / 3,002 |
| kindling TP=2, published | | 2,931 / 2,871 |

The profiles keep 8K chunks: 16K chunks halve the KV pool (above), and a longer chunk stalls decoding requests for
longer. A dedicated prefill workload can set `--max-prefill-tokens=16384` in its own profile.

**[RigMark](https://github.com/alexellis/rigmark)** (Alex Ellis) at `40fabcaf`, thinking on with low effort, 2 runs a
cell. The release image ran the full cell set of [bench/RIGMARK.md](../../bench/RIGMARK.md) on `8x128k`
(`raw/recipe/rig-*`, gates 6/6 and 6/6). An earlier dev build ran a short set on `4x512k` before the prefill switches
(`raw/rigmark/rig-f1.*`, gates 6/6).

| | Release image, `8x128k` | Dev build f1, `4x512k` | NVFP4, README (RC4, 2026-10-12) | NVFP4, shared engine (2026-10-15) |
|---|---:|---:|---:|---:|
| Decode, code / prose / structured (tok/s) | 67.9 / 37.0 / 90.8 (prose run: 67.7 / 37.1 / 91.0) | 69.1 / 39.1 / 94.9 | 72.3 / 38.3 / 91.7 | 69.1 / 39.3 / 91.9 |
| Cold prefill 8K / 32K / 64K: time to first token (s) | 3.21 / 11.89 / 22.67 | 3.29 / 12.45 / 23.80 | 3.29 / 12.47 / 23.91 | 3.29 / 12.48 / 23.83 |
| Staggered arrivals, prefill first: newcomer TTFT at 2 / 4 / 6 / 8 (s) | 2.64 / 3.36 / 4.08 / 4.75 | – | 2.80 / 3.54 at 2 / 4 | 2.81 / 3.54 at 2 / 4 |
| Short code, 1 / 2 / 3 / 4 streams aggregate (tok/s) | 47.8 / 60.6 / 68.6 / 72.2 | 53.2 / 60.2 / – / 71.1 | 47.0 / – / – / 70.9 | 47.7 / 58.8 / 62.7 / 72.8 |

On the release image, short code at 5 / 6 / 8 streams is 81.2 / 81.4 / 90.5 tok/s and short prose at 1 / 2 / 3 / 4 /
5 / 6 / 8 streams 34.6 / 46.4 / 55.0 / 58.5 / 60.9 / 64.1 / 72.1. With eight requests running, a 32K-token newcomer
waits 14.00 s for its first token when decoding goes first (1.18x its solo time; the incumbents stall up to 3.55 s);
with prefill first a short newcomer waits 4.75 s (6.79x solo) and the long one 1.44x solo.

Cold prefill is 2–5% faster than the README's RC4 figures, and staggered newcomers start 5% sooner. Single-stream
decode is lower: 6% on code, 3% on prose, 1% on structured. That is not an engine-only comparison: the
checkpoint changed, and QAD weights change how often the DFlash2 drafter's proposals are accepted, so decode speed
moves with the model as well as the kernels. RigMark samples, so each run writes a different program; its cells move by
several percent from start to start
([2026-10-15](../2026-10-15-shared-engine/RESULT.md#speed-same-prompts-production-against-this-release)).

**Mixed concurrency**, kindling `mixed4`/`mixed8`, `8x128k`, 8 batches (`raw/concurrency/`): 85.2 / 100.9 tok/s without the
prefill switches; median time to first token 2.44 / 4.67 s.

## Checkpoint-format (native) serving

`ATLAS_GLM_FP4L_NATIVE=1` serves every group in FP4-Local's own format, exactly: KDA `q/k/v/o` and the dense MLP from
their MXFP8 bytes, MLA and the shared experts from their FP8 block bytes. The default instead requantizes them into
production's formats (KDA and shared experts NVFP4, MLA MXFP8). Native decode reads about 1.49 GB more per rank per
step (KDA +1.07 GB). `4x512k`, kindling bench, 6 runs a workload (`raw/native/`):

| Arm | code / prose / structured (tok/s) | KV pool (blocks) |
|---|---:|---:|
| Production formats (`n-prod`) | 82.8 / 47.2 / 104.7 | 161,844 |
| MLA on its FP8 bytes (`n-mlaw16`) | 78.0 / 45.0 / 103.7 | 174,644 |
| MLA and shared experts on their FP8 bytes (`n-mlashw16`) | 80.7 / 45.3 / 101.2 | 199,132 |
| Every group native (`n-natw16`) | 72.6 / 39.7 / 94.6 | 204,170 |
| Every group native, tiled KDA twin (`n-natw16tile`) | 72.1 / 39.5 / 93.9 | 166,554 |

Fully native decode is 10–16% slower and scores the same on ToolEvalBench (91). It holds about 2.6 GB less per rank
before the KV pool (99.54 against 102.12 GB), so the pool grows by 26%. Greedy C1 and C4 agree 8/8 in every arm. The slowdown is the bytes, not the kernels: the fused tiled
MXFP8 KDA q/k/v GEMV takes 228.6 µs against 236.7 µs for a device copy of the same 51.9 MB, while production's NVFP4
reads 28.3 MB in 143.8 µs (`nbench-60f6.txt`, per rank, M=1). These arms prefilled native groups with W8A16
(`ATLAS_GLM_FP4L_FP8_PREFILL=w8a16`).

## Opt-in options measured

- **`ATLAS_DFLASH_WARM_CTX=restore`** (`raw/bench/warm-ctx/`): warm-turn decode 77.5 against 76.1 tok/s for `legacy`,
  one start each, the same greedy text; within run-to-run spread.
- **`ATLAS_GLM_PACKED_PREFILL=1`** with `ATLAS_TAIL_SPLIT_MIN_TOKENS=512` and `ATLAS_PREFILL_BURST=1`
  (`raw/concurrency/pk-on/`): mixed 86.4 / 105.7 tok/s, median time to first token 1.24 / 1.79 s against 2.44 / 4.67 s.
  Not yet exact against unpacked prefill, so it stays off.
- **`ATLAS_TAIL_SPLIT_MIN_TOKENS=512` with `ATLAS_PREFILL_BURST=1`** (`raw/concurrency/ad-burst/`): 84.7 / 104.4 tok/s,
  1.95 / 3.66 s.

## Release image

`ghcr.io/enntity/atlas-sparkglm:c8ced64316bf` (install tree `c8ced64316bf`, engine `b3b016ab`) was built by
`install/build.sh` from this branch on ennspark01 (warm BuildKit cache; all 14 build-time check stages passed,
`raw/recipe/build.log`), loaded on ennspark02 (the same image id, `29beb60fb7d2`) and published to GHCR (digest
`sha256:0b91fd790afb2ee3aa5a7f39a0bc5a90db05751436ae7810e00f129c36ebbb7b`). The first build of the branch's earlier
install tree stopped in the check stage: the FP4-Local unit tests read fixtures from the engine's `test_data/`, which
the Dockerfile now copies whole. Settings as in [Setup](#setup), on the shipped profiles (`raw/recipe/rel17.log` is
the driver's log):

| Check | Result |
|---|---|
| Start, `4x1m` | `atlas-recipe-start` names profile `4x1m` and source `Enntity/atlas@b3b016ab`; KV pool 181,120 blocks (`start-4x1m.rank0.log`) |
| Prompt logprobs of the 48,875-token text, two runs | `25bc828eea3f` both times, the dev gates' hash (`lp-glm-REL.json`) |
| `tool_probe.py`, 7 cases x 3 | 21/21 (`tools-REL.json`) |
| Tools + `response_format` probe, 5 runs, thinking off / on | 5/5 / 5/5 schema-valid final answers |
| ToolEvalBench, one trial, seed 42 | 91 (126/138; `teb-REL.log`, `teb-REL-*.md`), as in the three-trial dev runs (91.0 ± 0.0) |
| RigMark, `8x128k` (KV pool 144,464 blocks, `start-8x128k.rank0.log`) | [Speed](#speed); `rig-code-full.*`, `rig-prose.*` |

The checkpoint was already on disk, so this run did not exercise a fresh download.

**RigMark receipt metadata.** The receipts were first written from another product's metadata template, so
`run.appliance` named the Qwen checkpoint and its KV layout. Only those declared fields were corrected (the FP4-Local
checkpoint, the profile, the KV pool from each run's start log), and the card's JSON hash line updated to match:
`rig-code-full.json` `472b96fa…` → `a828cf5c…`, `rig-prose.json` `d996548e…` → `70f93321…`, and the dev
`rig-f1.json` `6fc337f9…` → `cbe13db6…`. Measurements, output hashes and gates are unchanged.

## Not covered

- A fresh download of the checkpoint through `./start.sh`, and its copy to the worker.
- The NVFP4 checkpoint through the new `CHECKPOINT_REPO`/`CHECKPOINT_REVISION` path end to end (the same binary served
  it through `start-node.sh`, above).
- Image prompts on FP4-Local's MXFP8/NVFP4 vision tower beyond loading.
- The 925K-token needle on FP4-Local.

## Receipts

`raw/` is curated: JSON and text results, profiles, and server logs trimmed to the load, the KV lines and every
warning or error. Host paths and fabric addresses are replaced (`~`, `RANK0_HOST`, `RANK1_HOST`).

- `gates/`: the release candidate's gates (`*-rc2*`), the g2 binary's FP4-Local (`N`) and NVFP4 (`O`) starts.
- `toolevalbench/`: one directory per arm with the tool's log, cross-trial summary and profile.
- `bench/`: kindling bench (`kb-f1.json`) with both profiles, and the warm-context arms (`warm-ctx/`).
- `recipe/`: the release image's build and start logs, prompt logprobs, tools, ToolEvalBench and RigMark.
- `rigmark/`: the dev build's RigMark card, JSON receipt and log.
- `prefill/`: one directory per cold-prefill arm: `prefill.txt`, prompt-logprob hash, profile, log summary.
- `concurrency/`: the mixed-stream arms.
- `native/`: the checkpoint-format arms and the GEMV microbenchmark `nbench-60f6.txt`.
- `harness/`: the scripts that drove the runs and the switch sets (`env/`). They call kindling's `prefill.py` and
  `gate/bench/bench.py` and SparkQwen's `quality_probe.py`, `tool_probe.py` and `greedy_eq.py`; kindling's code has no
  license and is not included here.
