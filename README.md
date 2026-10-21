<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<p align="center">
  <img src="docs/assets/banner.png" alt="SparkGLM: GLM-5.3-Flash on DGX Sparks" width="100%">
</p>

# SparkGLM

GLM-5.3-Flash on two or four NVIDIA DGX Sparks, served by the
[Atlas](https://github.com/Enntity/atlas) inference engine. One image supplies
TP2 and TP4, Strict and Spark modes, prefix caching, and an OpenAI-compatible
API. TP4 supports a switchless chain or an explicitly configured ring.
The supplied profiles allow up to four concurrent requests with a per-request
limit of the model's full 1M context; a TP2 profile also offers eight slots at 128K.

This unified release candidate is unpublished. Focused checks pass for Strict
and Spark on TP2 and TP4, TP4 chain/ring, native MTP near-1M shared-prefix
concurrency, and protected prefix-cache disk restore. Use the accompanying
source bundle and build instructions for this local candidate.
[Current status and receipts](results/2026-10-21-unified-rc/RESULT.md) identify
the tested configurations; historical tables below describe earlier releases.

```sh
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
cp .env.example .env      # set WORKER to the other Spark's ssh destination
./start.sh
```

## Strict and Spark modes

Select `MODE=strict` (default) or `MODE=spark` in `.env`, or run
`./start.sh --mode spark` for a TP2 pair. The per-node launcher accepts
`--mode strict|spark` on TP2 and TP4; managed containers use `SPARKGLM_MODE`.
Every rank must use the same image, profile and mode.

| Mode | Target execution |
|---|---|
| **Strict** | Preserves checkpoint weight formats and calibrated input scales, with A16 routed decode. Serial by default; the checkpoint's native MTP predictor is an explicit option. |
| **Spark** | Uses calibrated A4 routed decode and DFlash2, with corrected round-to-nearest-even FP4 KDA decode and shared-expert weights. KDA prefill, MLA, dense projections and the vocabulary head retain checkpoint formats. |

Drafter weight conversions use corrected rounding too. Spark disables
shared-expert TP splitting on TP2 and TP4 because the split's partial-output
rounding caused an arithmetic regression. The TP4 profile also disables it
in Strict after a matched C4 control corrected an arithmetic answer; TP2
Strict's split setting is unchanged. Unsupported weight formats fail
loading instead of silently converting. FP8 KV and prefill activation
quantization remain in both modes; Strict is not a promise of bit-identical
results across engines or prefill schedules.

For native MTP in Strict, pass `--mode strict --strict-mtp` to
`install/start-node.sh` on every rank, or set `SPARKGLM_MODE=strict` and
`SPARKGLM_STRICT_MTP=1` in every managed container. Use a profile with at most
four active/admitted requests (`4x1m`, `4x512k`, or `4x1m-tp4`). It retains
prefix caching and the profile's request-context limit. C4 near 1M with a shared
prefix does not mean four independent full-window contexts fit in memory.
`start.sh` currently exposes mode selection but not the native-MTP option.

Disk cache directories are separated by mode; TP4 also separates chain and
ring directories from each other and from TP2. TP4 sharded KV spill/restore
uses the same `--prefix-cache-dir` option. Resident-prefix TP4 Strict smoke
and two-cycle disk-restore checks passed. All release profiles protect matched
prefix blocks from being rewritten during another request's prefill. See the
[manual launcher instructions](docs/INSTALL-MANUAL.md#four-spark-chain-or-ring).

## Current RC measurements

On the TP4 ring, one matched run per workload and execution mode measured
the same engine before the final prefix write-protection setting:

| Workload | Strict native MTP (tok/s) | Strict serial (tok/s) |
|---|---:|---:|
| Prose | 41.531 | 39.888 |
| Code | 53.066 | 39.586 |
| Structured | 57.021 | 39.991 |

Output and reasoning hashes matched in all three pairs. These are focused
single-run comparisons, not a broad benchmark or a chain-versus-ring
comparison. A separate Strict ring capacity check passed with a
1,046,988-token prompt and four concurrent identical or distinct branches
of that shared prefix. It does not establish capacity for four independent
1M-token contexts. See the [full evidence and scope](results/2026-10-21-unified-rc/RESULT.md).

Spark on the final image, two repetitions per case with clocks uncapped:

| Workload | TP4 chain (tok/s) | TP4 ring (tok/s) | Ring change |
|---|---:|---:|---:|
| Prose | 57.145 | 58.328 | +2.07% |
| Code | 104.147 | 106.368 | +2.13% |
| Structured | 137.936 | 141.363 | +2.48% |

These are medians of the two runs. Every paired output, reasoning and stream
hash matched. All requests completed naturally; all repetitions and the
matched configuration proof are in the linked report. Two runs per case do
not establish statistical significance.

## What it does

Historical measurements on two DGX Sparks joined by one 200G cable. Rows
marked † are the 2026-10-17 FP4-Local release: RigMark and ToolEvalBench on its image
`c8ced64316bf`, the other † rows on dev binaries of the same engine. The
unmarked rows ran images built from a fresh clone with the NVFP4 checkpoint. Receipts and caveats:
[2026-10-17: FP4-Local, ToolEvalBench, kindling's benchmarks](results/2026-10-17-fp4local/RESULT.md),
[the shared engine](results/2026-10-15-shared-engine/RESULT.md),
[RC4: faster single-stream decode, bit-exact with RC2](results/2026-10-12-rc4/RESULT.md),
[RC2: KV shard, deterministic verify, faster MoE decode](results/2026-10-09-rc2/RESULT.md),
[the display carveout's KV placement](results/2026-10-05-carveout-placement/RESULT.md),
[a faster verify step and cheaper warm turns](results/2026-09-30-decode-step/RESULT.md),
[RigMark decode, prefill and staggered arrivals on 2026-10-05](results/2026-10-05-rigmark/RESULT.md),
[RigMark short-code concurrency](results/2026-09-30-rigmark/RESULT.md),
[prefix cache on disk](results/2026-09-30-nvme-tier/RESULT.md),
[exact kernels and the prefix-cache policy](results/2026-09-30-exact-speedups/RESULT.md),
[index tails and 0.92 memory](results/2026-09-29-index-tails-092/RESULT.md),
[prefix caching](results/2026-09-29-prefix-caching/RESULT.md)
and [the first Atlas release](results/2026-09-29-atlas-merged/RESULT.md)
(quality row).

| Workload | SparkGLM (Atlas) | Reference, same pair |
|---|---|---|
| Multi-turn conversation, 30–48K tokens: time to first token on turns 2+ (median) | **0.67 s** | 17.6 s with caching off |
| Four concurrent ~204K-token sessions, cached follow-up turns † | 12/12 numbers retrieved (two cut to 3 digits by the probe's token cap) | |
| KV pool shared by all requests (0.91, display carveout, disk prefix cache on) † | **2.66M tokens** (2.90M on the release image's start), up to 1M per request | 2.34M with the NVFP4 checkpoint, same engine |
| Same greedy output at 1 and 4 concurrent requests † | **8/8 prompts** | 0/4 on 2026-10-09 morning |
| Replaying a 35K-token prompt † | **0.38 s** (13.4 s cold) | |
| New session sharing a 24K system prompt with earlier ones: time to first token | **15.7 s** | 36.2 s with the cache policy off |
| Idle session resuming after another session's 17 turns (median) | **2.5 s** | 49.6 s with the cache policy off |
| Cold prefill, 32K / 128K random words (kindling's `prefill.py`) † | **2,645 / 2,733 tok/s** (2,886 / 3,002 with 16K chunks) | [kindling](https://github.com/kindlingai/glm-5.3-flash-gx10) vLLM TP=2 (published) 2,931 / 2,871 |
| Matrix: C1/C2 at 16K and 32K, C4 at 16K, 400 tokens each, cold (sum of walls) | **132.1 s** | vLLM SparkGLM 206.6 s · Mia TensorFold v1.10 166.9 s |
| Staggered C4: four ~16K requests arriving 1 s apart, cold (median of 5) | **49.5 s** | vLLM SparkGLM 70.5 s · Mia TensorFold v1.10 60.2 s · Mia EXL3 113.2 s |
| Staggered C4: four ~32K requests arriving 1 s apart, cold (median of 3) | **76.3 s** | Mia TensorFold v1.10 100.1 s |
| Single-stream decode, kindling's `bench.py`, code / prose / structured / counting (thinking off, 16 prompts each) † | **72.8 / 44.4 / 99.5 / 112.2 tok/s** | kindling vLLM TP=2 (published) 66.2 / 42.6 / 94.7 / 97.2 |
| Mixed streams, kindling's `bench.py`, 4 / 8 at once, aggregate † | **82.6 / 100.6 tok/s** | kindling TP=2 (published) 82.6 / 109.6; 89 / 116 remeasured |
| RigMark decode, code / prose / structured (thinking on, low effort) † | **67.9 / 37.0 / 90.8 tok/s** | [RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE) vLLM TP2 (published) 56.5 / 33.0 / 83.4 · NVFP4 checkpoint (RC4) 72.3 / 38.3 / 91.7 |
| RigMark cold prefill 8K / 32K / 64K: time to first token † | **3.21 / 11.89 / 22.67 s** | RiNGSiDE 3.56 / 12.94 / 25.66 s · RC4 3.29 / 12.47 / 23.91 s |
| RigMark staggered arrivals, prefill first: newcomer time to first token at 2 / 4 † | **2.64 / 3.36 s** | RiNGSiDE 4.70 / 5.15 s · RC4 2.80 / 3.54 s |
| RigMark short code, 1 / 4 / 6 streams, aggregate † | **47.8 / 72.2 / 81.4 tok/s** | RiNGSiDE 44.0 at 1 stream, 97.3 at 6 · RC4 47.0 / 70.9 at 1 / 4 |
| Returning to a ~209K-token conversation evicted to disk (prefix cache on disk, 48 GB) | **1.2–1.5 s** | about 82 s without it |
| Strict JSON (`response_format` json_schema), ~2,000-token structured answer | **30–50 s**, schema-valid | 135–155 s with masked serial decode |
| Strict JSON object with 96 required keys, temperature 0 | **39 s**, 59 tok/s, valid | engine killed (out of memory) two releases ago |
| ToolEvalBench, 69 scenarios × 3 trials, temperature 0 † | **91.0** (91 in one trial on the release image) | NVIDIA's model card: FP4-Local 92.4, BF16 91.9 (an earlier 63-scenario version) |
| Quality probe: arithmetic / two-hop 24K needle | 39/40 · 12/12 † (the same as with NVFP4); 491/500 · 24/24 on the 500-item set (NVFP4) | every engine configuration we ran scored 484–495/500 |

These dated measurements do not describe the unified TP2/TP4 RC. Rows
not re-run on 2026-10-17 keep their earlier receipts linked above. Against RC4 on the NVFP4
checkpoint, FP4-Local prefills 2–5% faster and RigMark decode is 1–6% slower:
the QAD weights change how often the drafter's proposals are accepted, so
decode speed is not an engine-only comparison
([details](results/2026-10-17-fp4local/RESULT.md#speed)). Kindling's figures are for
their vLLM recipe with the NVFP4 checkpoint; ours ran their benchmark scripts
against FP4-Local ([details](results/2026-10-17-fp4local/RESULT.md#speed)). Our
profiles prefill in 8K chunks: 16K chunks are faster but halve the KV pool. Known gaps and open questions
are in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Requirements

- Two DGX Spark (GB10, 128 GB) systems joined by direct ConnectX-7 links,
  or four systems connected as a switchless chain or ring. Each selected
  RDMA rail needs an IPv4 RoCE address and a matching link to its neighbour.
  A dual-rail ring needs both rails on the closing rank 3–rank 0 link too.
- Docker with the NVIDIA runtime on every node. The TP2 `start.sh` launcher
  also needs passwordless `ssh` to its worker, whose user can run `docker`.
- About 190 GB free on each Spark: the checkpoint takes 177.5 GB (165.3 GiB)
  and the drafter 2 GB. The download is made once and copied to the other
  Spark over the cable. No on-disk conversion is needed for FP4-Local. (The earlier NVFP4 checkpoint,
  which `.env` can still select, takes 204 GB plus a 16 GB overlay.)
- Nothing else using the GPUs' memory. GB10 memory is shared with the host,
  and the engine uses most of it.

## What `./start.sh` does

Run it on the Spark that will serve the API (rank 0). Each step is skipped when
already done, so rerunning it just restarts the engine.

`start.sh` manages a TP2 pair. TP4 uses the same image and
`install/start-node.sh --world-size 4 --profile 4x1m-tp4` on each rank, or
[LLooM](docs/LLOOM.md) four-rank orchestration. Chain is the default; ring is
selected explicitly with `--topology ring` and requires the closing links.
[Manual TP4 setup](docs/INSTALL-MANUAL.md#four-spark-chain-or-ring) describes
the per-rank rails.

1. **Image.** Pulls `ghcr.io/enntity/atlas-sparkglm:<tag>`, whose tag is the
   git tree of [`install/`](install/), so the image always matches your
   checkout. If it isn't published or `BUILD=1` is set, it builds the image
   from source instead ([`install/build.sh`](install/build.sh): Atlas at the
   commit pinned in [`install/atlas-source.json`](install/atlas-source.json),
   FlashKDA and FlashInfer's sparse-MLA prefill; 30–60 minutes cold). It then
   pulls or copies the image to the other Spark.
2. **Weights.** Downloads [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)
   and the [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
   drafter at pinned revisions, and copies them to the other Spark. It uses
   `hf` when installed and a throwaway container otherwise.
3. **Overlay, NVFP4 checkpoint only.** FP4-Local needs no on-disk overlay. If
   `.env` selects the NVFP4 checkpoint ([below](#the-checkpoint)), this step
   converts the 864 BF16 MTP matrices Atlas needs to NVFP4 in an overlay on
   each Spark and verifies every payload (about a minute;
   [`install/convert.sh`](install/convert.sh)).
4. **Serve.** Starts rank 1 over ssh, then rank 0, and waits for health
   (about 2.5 minutes), then answers one smoke-test question.

Other commands: `./start.sh stop | status | logs [worker] | build | download | convert`.

## Use it

The API listens on rank 0's loopback, `http://127.0.0.1:8893/v1`, model
`glm-5.3-flash-atlas`, with no authentication. Put your own authenticated
proxy in front of it before exposing it.

```sh
curl -s http://127.0.0.1:8893/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash-atlas", "max_tokens": 2048,
  "messages": [{"role": "user", "content": "Write a haiku about two small computers."}]}'
```

Streaming, tool calls, JSON and structured output, images and short videos
are supported. GLM's checkpoint template opens reasoning even with
`chat_template_kwargs.enable_thinking: false`; its default is Max effort,
with low/high selected through `reasoning_effort`. The candidate no longer
adds a short reasoning cap. `max_tokens` includes reasoning and can end a
request before the answer appears; an explicit thinking-token budget can
cap the reasoning section. See [behavior limits](docs/LIMITATIONS.md#behaviour).

For serial/speculative comparisons, `/v1/completions` accepts
`disable_mtp: true`. Blocking completions can request `return_token_ids: true`
for exact generated IDs in `choices[].token_ids`; streaming with this option
is rejected. Both default to false. [Engine details](docs/ENGINE.md#candidate-behavior)
explain their limits.

When an agent repeats the same tool call and gets the same result, the engine
adds a repeat notice to that result and lets the model think harder on that
turn; it no longer blocks tool calls, which had left the model narrating the
action it meant to take ([RC4.2](results/2026-10-12-rc4/RESULT.md)).

**Profiles** (`PROFILE` in `.env`):

- `4x1m` (default): up to four requests with up to 1,048,576 tokens each, the
  model's full window.
- `4x512k`: up to four requests with up to 512K each.
- `8x128k`: up to eight requests with up to 128K each, for more concurrent
  short work.
- `4x1m-tp4`: the TP4 profile, selected by four-rank orchestration
  rather than `PROFILE` in `start.sh`; four sequences, up to 1M tokens per
  request, with memory utilization 0.72. See the
  [TP4 limits](docs/LIMITATIONS.md#four-spark-tp4-candidate).

The context limit applies to each request; it does not reserve that much KV
for every concurrent slot. At the historical TP2 2.66M-token pool size, four full
1M-token requests cannot fit together. Requests wait when the pool lacks room.

All requests share one FP8-latent KV pool that also holds the prefix cache.
The figures in this section are historical TP2 measurements; candidate
capacity must be read from its own startup and load-test receipts.
For the TP2 launcher, the KV cache is split between the two Sparks (`KV_SHARD`, on by default): each
stores the latents of half the blocks, so the pool holds about 1.8x the tokens,
and the output is bit-identical to the unsplit pool's. On Sparks that run
nothing else, at `GPU_MEMORY_UTILIZATION=0.91` with the display carveout and
the disk prefix cache (both below), the pool is about 2.66M tokens with
FP4-Local (2.5–2.7M across starts). Four concurrent 204K-token sessions left
rank 0 at 3.3 GB free at worst. Above
0.91, real traffic took our rank 0 below 1 GiB free
([docs/LIMITATIONS.md](docs/LIMITATIONS.md)). Requests that don't fit wait for
room. Prefix caching keeps the 16 most recent conversations' recurrent state
warm.

## The checkpoint

[`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)
@ `76348ddb` is NVIDIA's quantization-aware distilled release: routed experts
in NVFP4 with losslessly compressed block scales, the KDA projections, dense
MLP and `lm_head` in MXFP8, the MLA projections and shared experts in FP8 with
128x128 block scales, and the MTP layer and vision tower already quantized. It
is 25 GiB smaller than NVIDIA's earlier NVFP4 release, so the KV pool gains
about 13%, and it needs no overlay.

The unified RC selects precision through [Strict and Spark modes](#strict-and-spark-modes).
Both use checkpoint input scales. Strict keeps native projection weights and
A16 routed decode; Spark permits the selected FP4 KDA/shared-expert conversions
with corrected rounding and A4 routed decode. MLA, dense projections and the
vocabulary head remain native in both. These policies replace the earlier
blanket requantization and the intermediate native-format-only candidate.
The [current RC evidence](results/2026-10-21-unified-rc/RESULT.md) is separate
from the 2026-10-17 historical measurements, which do not describe
these mode presets. See [fidelity limits](docs/LIMITATIONS.md#fp4-local).

The installer retains its earlier NVFP4 checkpoint override. It is outside
the unified Strict/Spark RC qualification; required native-format checks still
apply at load. Prior prompt-logprob comparisons are historical.
To select it, add both lines to `.env` and rerun
`./start.sh`; it downloads the checkpoint and converts its overlay:

```sh
CHECKPOINT_REPO=nvidia/GLM-5.3-Flash-NVFP4
CHECKPOINT_REVISION=423acf37583782c51c142d145aef733d72943d93
```

## Prefix cache on disk (optional)

TP4 now supports sharded KV spill/restore through
`install/start-node.sh --kv-shard --prefix-cache-dir DIR` on every rank,
with separate TP, topology and mode namespaces. The
[TP4 Strict storage check](results/2026-10-21-unified-rc/RESULT.md) passed two
eviction/restore cycles on all four ranks with exact cold/warm native/serial
outputs. It used a 2,048-block diagnostic GPU pool to force eviction and the
normal 48 GiB combined disk tier. The measurements and capacity figures
below are historical TP2 results.

When the KV pool fills up, the prefix cache drops the conversations used
least recently, and one that comes back is prefilled again from the start.
With `PREFIX_CACHE_DIR` set, each Spark writes what it drops to its own disk
instead and reads it back when the conversation returns. In one pass on
2026-09-30, with 25K-token conversations pushed out of the cache, the time to
first token of turn 2 went from 10.7 s to 0.91 s and the answers were
byte-identical. Each restore read its blocks in about 75 ms (7.8 GB/s). With
the KV shard (the default) each Spark spills and restores only the latents it
owns: on 2026-10-09 a 25K-token conversation came back in about 40 ms per
restore (5.2 GB/s), and turns 2-4 after three restores were byte-identical to a
run that never evicted, token logprobs included.

To turn it on, add to `.env` and rerun `./start.sh`:

```sh
PREFIX_CACHE_DIR=/srv/sparkglm/prefix-cache   # same path on both Sparks, on their own disk
PREFIX_CACHE_GB=48                             # GiB per Spark, 16-100 (default 48)
```

`start.sh` creates `kv/` and `ssm/` in the directory on both Sparks, refuses a
directory on tmpfs or without that much space free, and mounts it into each
rank. Half of the size holds KV blocks (6.66 KB per token unsplit, 3.7 KB per
Spark with the KV shard; 24 GiB holds 3.9M or 6.9M tokens). The other half holds the 78 MB
recurrent-state snapshots a restore starts from (24 GiB holds 330). A restore
needs both, so at 25K tokens the even split keeps about 150 conversations with
two snapshots each.

What it costs, per Spark:

- **KV pool.** The engine sets host memory aside for the tier and takes it
  out of the pool: 236 MiB whatever the size (staging, and two snapshot
  slots), plus 640 B for each 16-token block the KV half can hold, which is
  6.2 MiB per GiB. At the default that is 383.8 MiB. On our pair at 0.93 the
  pool lost about 4,000 of its 104,000 blocks, about 60K tokens (4%); at the
  default 0.88 the same 60K tokens are about 10% of the pool. With the KV
shard each block costs each Spark about half as much memory, so the same
reservation takes about twice the tokens: at 0.91 with the carveout the pool
goes from 163,177 to 147,480 blocks (2.61M to 2.36M tokens). Each GiB added
  to `PREFIX_CACHE_GB` costs about 500 more tokens. The fixed part is most of
  the cost, and 48 is the size we measured.
- **Disk.** The KV half is reserved when the engine starts; the snapshot half
  fills as needed. Every token pushed out of the pool writes 6.66 KB. Nothing
  is kept across a restart: the engine deletes its files as soon as it has
  them open, and clears leftovers from a crash when it starts.

## Display memory as KV cache (optional, DGX Spark only)

Each DGX Spark's firmware sets aside 2 GiB of memory for the display (the
`DISPLAY_FRM` carveout). Linux never sees it; the NVIDIA driver manages it as
the display's own memory pool, and on a headless Spark the display uses only
a few MiB of it. With `DISPLAY_CARVEOUT=1`, the engine asks the driver for
the rest (about 2042 MiB) and borrows it for the KV cache: it places whole
per-layer latent KV pools there, so the pool grows while system memory stays
exactly as it was. Because the driver hands it out, it never overlaps what
the display uses, and the GPU caches it like ordinary memory
([results/2026-10-14-vision-prefix-cache](results/2026-10-14-vision-prefix-cache/RESULT.md)).

On our pair at 0.91 GPU memory utilization, the pool grows from 70,501 to
86,005 blocks per Spark (+19.4%, about 1.13M to 1.38M tokens). Cold prefill
at 32K and 64K stays within 0.7% of the carveout off, and greedy outputs are
unchanged.

To turn it on, add to `.env` and rerun `./start.sh`:

```sh
DISPLAY_CARVEOUT=1   # default off; set 0 or delete the line to roll back
```

What it takes:

- **No extra container privilege.** The engine starts through
  `spark display-carveout`, which allocates the memory from the driver, takes
  a host lock and then starts the server.
- **One user per Spark.** The lock in `/run/lock/sparkglm` lets one process
  per host hold the carveout. `start.sh` creates the directory.
- **Driver.** Every NVIDIA open GPU kernel module release from 580.76.05 to
  615.78.08 provides it; the engine checks what the driver gives rather than
  its version. If the driver cannot lend it, the engine serves without the
  carveout and logs why.
- **A headless Spark.** The engine takes what the display is not using when
  it starts; a display attached later may find no room for its surfaces.
  We run ours headless.

This is SparkGLM-only: it ships in the SparkGLM layer of the engine fork, not
in the upstream Atlas series. The idea comes from kindling-spark-os's
`dispram` (see [Credits](#credits)).

## Reproduce our numbers

The benchmark drivers are in [`bench/`](bench/). See
[docs/REPRODUCE.md](docs/REPRODUCE.md) for the exact commands and the
comparison videos.

## Other ways to run it

- **By hand:** every step of `./start.sh` as a separate command, in
  [docs/INSTALL-MANUAL.md](docs/INSTALL-MANUAL.md).
- **With [LLooM](https://github.com/Enntity/lloom):** the recipe
  `linux-nvidia-dgx-spark-2x-glm53-atlas` puts SparkGLM behind LLooM's
  authenticated gateway ([docs/LLOOM.md](docs/LLOOM.md)).

## The engine

Atlas here is built from [`Enntity/atlas`](https://github.com/Enntity/atlas)
at the commit and tree in
[`install/atlas-source.json`](install/atlas-source.json). The 2026-10-19
candidate adds TP4, checkpoint behavior fixes and sampled verification to
the earlier `sparkglm/atlas-20261017-fp4local` foundation at `b3b016ab`.
Its behavior and switches are described in [docs/ENGINE.md](docs/ENGINE.md). The fork keeps one
branch per layer: `spark-shared`, which SparkQwen builds on too, and `sparkglm`
on top of it ([branch and release index](https://github.com/Enntity/atlas/blob/enntity/releases/docs/PUBLISHED_RELEASES.md)).
The engine is built from three layers:

1. Atlas-Inf `main` at `7ffd183f`.
2. The shared Spark engine layer `upstream/atlas-shared-20261015-vision` at
   `a9464810`, which [SparkQwen](https://github.com/Enntity/sparkqwen) builds on too. It holds
   GLM-5.3-Flash support and optimizations, which we intend to propose to
   Atlas-Inf after review (it started as Reiner Schmidt's port,
   [Mango-kid/atlas](https://github.com/Mango-kid/atlas/tree/feat/glm53-dual-spark);
   the engine's `docs/porting/GLM_5_3_FLASH.md` has the history); the
   Qwen3.8-Flash-Next series, whose paths stay off for GLM; and the
   model-independent work both products share: speculation that commits
   exactly the tokens plain greedy decode commits (one lowest-index tie rule),
   shared-prefix admission, opt-in content-loop steering, batched prefix-cache
   eviction, the refusal-content fix (Atlas-Inf issue 172), the vision
   prefix cache below and the vision fixes both models share: every vision
   tower encodes per chunk into per-sequence state, a malformed vision
   layout fails only its request, MRoPE positions come from the whole
   prompt, and vision markup typed in message text encodes as text.
3. SparkGLM-only commits: FlashKDA and native sparse-MLA prefill bridges,
   which rely on libraries built outside the Atlas tree, the GB10 display
   carveout (including where the split KV pools are placed in it), and the
   FP4-Local series: the compressed-scale decoder and loader, four exact
   prefill options the profiles turn on, the tools + `response_format` union
   grammar and the checkpoint-format path, now enabled by product profiles.
   The candidate adds calibrated routed input scales, experimental A16 routed
   decode/verify (disabled in product profiles), FP32 router logits, checkpoint prompt/reasoning behavior, sampled verify
   and the TP4 chain. Its model-independent
   commits (`shared:`) are candidates for the shared layer.

The profiles set `ATLAS_MOE_INPLACE_TRANSPOSE=0`: the shared engine's default
in-place MoE transpose keeps every expert's allocation, which on GB10 costs
about 1.5 GB of kernel memory per Spark that the KV pool would otherwise get.
In the 2026-10-15 comparison against its predecessor, text prompts scored
the same prompt logprobs and the prefix-cache, fill, tool-call and vision checks passed
([results/2026-10-15-shared-engine](results/2026-10-15-shared-engine/RESULT.md)).

**Image prompts use the prefix cache.** Every image used to be the same run
of placeholder tokens, so image prompts bypassed the prefix cache entirely:
once an image entered a long conversation, every turn re-prefilled the whole
history and re-encoded every image. The cache key now replaces each
placeholder with an id derived from its image's SHA-256 content hash, so a
different image never shares KV and the same image hits; text before the first
image keys as before. Only images a prefill actually computes go through the
vision encoder, the second Spark no longer loads the vision tower (rank 0
encodes and sends the rows), and concurrent image prompts keep their vision
state per request. In that 2026-10-14 comparison, text prompt-logprob hashes
were unchanged. Receipts: [results/2026-10-14-vision-prefix-cache](results/2026-10-14-vision-prefix-cache/RESULT.md).

The manifest's `layers` entries distinguish the published historical
foundations from the qualification candidate. A source pin and passing
source tests do not establish image publication or hardware qualification.

## History

SparkGLM started as an Atlas research project: GLM-5.3-Flash on two DGX
Sparks with the Atlas engine. vLLM was far ahead at the time, so we moved to
it. On vLLM we started from an early version of
[MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)'s
two-Spark EXL3 recipe and reworked it, mostly for concurrency. We then
decided there was a lot we needed to fix at the engine level, and went back
to Atlas. Atlas is now the only engine.

The vLLM version is retired. It is preserved at the tag
[`vllm-final`](https://github.com/Enntity/sparkglm/tree/vllm-final) and the
branch `archive/vllm`, including its results and qualification records.

## Credits

The engine options SparkGLM turns on take their ideas from other projects. The
code is ours unless [docs/LICENSING.md](docs/LICENSING.md) says otherwise.

| Option | Idea from |
|---|---|
| `ATLAS_DFLASH_CONF_WIDTH` | knapcio's draft-shape truncation, `GLM_DRAFT_TRUNC` ([knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)), whose calibration table (MIT) seeds ours; and the survival prefix product of D-Cut ([arXiv 2607.14647](https://arxiv.org/abs/2607.14647)) as Atlas-Inf implements it for MTP. We added online calibration, a fixed row price and a periodic full-width probe. |
| `ATLAS_GLM_DRAFT_TP` | MiaAI-Lab's `DFLASH_DRAFT_TP` ([GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)) and TensorFold's two-rank drafter ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)) |
| `ATLAS_GLM_PC_EVICT` | Reederey87's prefix-cache eviction policy ([glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)) |
| `ATLAS_GLM_PC_BRANCH` | Marconi's branch-point admission (Pan et al., MLSys 2025, [arXiv:2411.19379](https://arxiv.org/abs/2411.19379)) |
| `ATLAS_GLM_STRICT_SPEC` | vLLM's speculative decoding with structured outputs ([#14702](https://github.com/vllm-project/vllm/pull/14702)) and its reasoning-boundary fix ([#44297](https://github.com/vllm-project/vllm/pull/44297)): per-position grammar masks over the draft window, the bonus row included, with the matcher rolled back after rejected drafts. We added per-rank masking for the vocab-split verify head. |
| `ATLAS_GLM_CANONICAL_VERIFY` | Batch-invariant kernels, from Thinking Machines Lab's "Defeating Nondeterminism in LLM Inference" (Horace He, 2025): a row's result must not depend on the rows computed beside it. We apply it to every op a DFlash verify row touches, at every verify width. |
| `ATLAS_KDA_FLASH_PREFILL` (FlashKDA's FP32 recurrence) | vLLM's FlashKDA fork keeping the recurrent state in FP32 fragments between tiles, narrowed to BF16 only for the MMA operand ([vllm-project/FlashKDA#13](https://github.com/vllm-project/FlashKDA/pull/13), Matt Mastracci). We also keep the state FP32 across calls, split every MMA operand into BF16 hi/lo pairs and compute the gate, beta, inverse and sums in FP32, to reduce deviation from the FP32 token recurrence; the exact native library and model-level behavior require their own qualification. |
| `DISPLAY_CARVEOUT` | kindling-spark-os's `dispram` ([kindlingai/kindling-spark-os](https://github.com/kindlingai/kindling-spark-os), by mmastrac, coffee-the-dev and adapt-ai-systems; they credit an earlier NVIDIA developer forum post by emihuang). Our implementation was re-derived from the open GPU kernel modules and needs no daemon. |

Measurement: [RigMark](https://github.com/alexellis/rigmark) by Alex Ellis
(run from othexmr's fork with the staggered-arrival suite), the published
[RiNGSiDE](https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE)
figures, [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10)'s
`bench.py` and `prefill.py` and its published figures,
[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) by
SeraphimSerapis, [mmastrac](https://github.com/mmastrac/glm-5.3-flash-4x-gx10)'s
published figures and benchmark method (for earlier receipts), and MiaAI-Lab's
decode prompts.

## License

AGPL-3.0-only ([LICENSE](LICENSE)); third-party components keep their own
licenses ([NOTICE](NOTICE), [docs/LICENSING.md](docs/LICENSING.md)). The model
weights are not distributed here. FP4-Local is under the NVIDIA Open Model
Agreement. **The DFlash2 drafter is licensed CC BY-NC-ND 4.0
(non-commercial).** Check both checkpoints' terms before downloading.
