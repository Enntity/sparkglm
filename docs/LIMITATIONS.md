<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Limitations

What to know before relying on SparkGLM.

## Scope of the measurements

- Historical comparisons used one pair of DGX Sparks on one cable. The
  unified RC also has four-Spark chain/ring checks. Other firmware, drivers,
  cables or cooling may give different numbers.
- The vLLM and Mia comparison runs were recorded earlier on the same pair.
  They were not alternated with the Atlas runs.
- Each README row comes from the dated release its receipt names. The
  unified TP2/TP4 RC has a separate
  [status and evidence page](../results/2026-10-21-unified-rc/RESULT.md);
  earlier component, development-binary and release measurements do not
  establish its performance or quality at TP2 or TP4. RigMark decode,
  cold prefill, staggered arrivals, the staggered C4 field guide, the Matrix
  and the four-session 204K fill were re-measured on the historical release,
  2026-10-09 RC2. RigMark short-code concurrency was also re-measured on RC2
  (47.7 tok/s at one stream, 69.5 at four); the earlier 2026-09-30 receipt is
  kept for history. The quality probe's first recorded run is from the first
  Atlas release, but the 500-item arithmetic (491/500) and two-hop 24K needle
  (24/24) sets were re-run on RC2.
- A cold image build has not been timed. Our builds reused BuildKit caches.
- Of the 2026-10-17 FP4-Local rows (marked in the README), RigMark, the prompt
  logprobs, tool calls and ToolEvalBench ran on the release image; the gates,
  the KV-pool comparison and kindling's benchmarks ran on dev binaries of the
  same engine on the previous image. RigMark runs two runs a cell.
- RigMark decode on FP4-Local is 1–6% below RC4's on the NVFP4 checkpoint. The
  weights differ, and with them the drafter's acceptance; we have not
  separated that from engine effects.

## Not yet measured

- **Independent full-window contexts:** the Strict ring check passed at a
  1,046,988-token prompt with four concurrent branches sharing that prefix.
  It does not establish capacity for four independent million-token contexts.
- **Chunked prefill and endurance:** prefill is not row-invariant. A prompt
  long enough to prefill in several chunks (over 8K tokens) can still differ
  when the engine is busy. Focused cancellation/reuse checks passed;
  sustained long-context endurance and multimodal work at full context are
  not measured.

## Capacity and prefix caching

The bulleted pool sizes and memory margins below are historical TP2 measurements.
The unified RC selects precision by mode: Strict uses native weights and A16
routed decode, while Spark uses the selected corrected FP4 conversions and
calibrated A4 routed decode. Read the
[current pool and memory receipts](../results/2026-10-21-unified-rc/RESULT.md)
before planning capacity.
Native MTP has a private predictor cache on rank 0; near 1M shared-prefix C4
must not be interpreted as four independent full-window contexts.

The Strict TP4 ring capacity check used a 1,046,988-token prompt. Identical
and distinct branches reached decode width four; the four distinct resident
branches overlapped for 8.539699 seconds. Minimum observed host
`MemAvailable` was 25.209 / 45.570 / 39.974 / 45.678 GiB on nodes 01 / 02 /
03 / 04. These are margins observed in that run, not reserved memory or an
endurance guarantee.

- All requests share one KV pool, which also holds the prefix cache. With the
  NVFP4 checkpoint, unsplit, it is about 600K tokens at the default 0.88 memory
  utilization and about 1.13M at 0.91 on dedicated Sparks. With the KV shard
  (`./start.sh`'s default) it holds about 1.8x that: 2.61M tokens at 0.91 with
  the display carveout, or 2.36M with the disk prefix cache. FP4-Local leaves
  more room: 2.66M tokens with the carveout and the disk prefix cache on the
  same binary that gave the NVFP4 checkpoint 2.34M (2.5–2.7M across starts). A request needs pool room for its whole
  context, so four full 1M requests do not fit at once; later requests wait
  for room.
- At 0.93 on dedicated Sparks the pool is about 1.63M tokens, and rank 0 went
  down to 1.35 GB of available memory at the end of a four-session 204K fill.
- At 0.92, a Spark sharing the host with other services went down to 3.1 GB of
  free memory. GB10 hosts can hang when memory runs out, so raise the setting
  only on Sparks that run nothing else.
- Under real mixed traffic at 0.93, rank 0 dropped below our 1 GiB memory guard
  about hourly (2026-10-02 to 10-04), so our production setting is 0.91. At
  0.91 the pool is about 1.13M tokens unsplit, or about 1.38M with
  `DISPLAY_CARVEOUT=1`, which adds KV without using system memory. On the
  2026-10-09 RC2 release, with the KV shard, the carveout and the disk prefix
  cache, the pool is about 2.36M tokens and a four-session 204K fill left
  rank 0 at 4.8 GB free at worst. With FP4-Local (2026-10-17) the same fill
  left 3.29 GB at worst: the larger pool takes the memory the smaller
  checkpoint frees. Only the latent KV pools go to the carveout; they already
  fill it.
- `DISPLAY_CARVEOUT=1` needs an NVIDIA driver that lends the display memory
  (open kernel modules 580.76.05 or later) and a headless Spark. It has been
  measured on our two pairs only.
- Prefix caching keeps 16 recurrent-state snapshots, one per recent request.
  A conversation idle for longer than that is re-prefilled from its deepest
  remaining cached block, or from scratch.
- Only block-aligned snapshots written during prefill are restored
  (`ATLAS_MARCONI_PREFILL_ONLY`). The engine's other snapshot kinds gave a
  wrong retrieval answer on a cache hit; see the prefix-caching result.
- Cache matches stop at the last whole 16-token block
  (`ATLAS_PREFIX_SUBBLOCK=0`). Before 2026-09-30 the profiles also let a
  request share a cached block that was only partly filled. Two requests could
  then write the rest of the same block, for example a retry of an identical
  prompt or the second of `n > 1` choices, whenever the prompt length was not
  a multiple of 16. GLM restores from block-aligned snapshots anyway, so
  turning this off costs nothing.
- The prefix cache on disk (`PREFIX_CACHE_DIR`, off by default) has been
  timed on historical TP2 25K-token conversations; the current TP4 check
  adds two exact restore cycles under a bounded diagnostic pool. Disk restores
  near the context limit, many simultaneous restores, eviction latency imposed
  on another request, and endurance are not measured.
- The engine refuses to start when the directory is on tmpfs, ramfs or
  overlayfs, cannot be written, or cannot hold the KV half of the size, and
  when the ranks' settings differ. Its host memory comes out of the KV
  pool (see the README).
- With caching on, the engine disables its fused prefill/decode path. Cold
  C4-16K is about 6% slower than in the release without caching.
- The first start after installing or updating compiles CUDA kernels. On our
  pair this briefly used about 2 GB more host memory than steady state.

## FP4-Local

- Strict preserves checkpoint weight formats; Spark intentionally converts
  KDA decode and shared-expert weights to FP4 with corrected nearest-even
  rounding. MLA, dense projections, the vocabulary head and KDA prefill stay
  native. Spark shared-expert TP splitting is disabled; the TP4 profile also
  disables it in Strict after the matched C4 arithmetic control. TP2 Strict
  is unchanged. Older blanket
  requantization and native-format-only timings do not describe either mode.
- Both modes use checkpoint input scales. Strict uses A16 routed decode and
  Spark uses calibrated A4. Weight-format preservation does not remove
  activation/KV quantization, floating-point rounding, prefill chunk effects
  or the need for matched model-level quality and sampling tests.
- The published 2026-10-17 FP4-Local evidence checked its MXFP8/NVFP4 vision
  tower only as far as loading. The candidate's vision changes need their
  own image, conversation and caching receipts.
- The Matrix, the field guide and the 925K-token needle have not been re-run
  on FP4-Local; those README rows are from the NVFP4 checkpoint.
- Serving the NVFP4 checkpoint through `CHECKPOINT_REPO`/`CHECKPOINT_REVISION`
  has not been run end to end through `./start.sh`; the same engine served it
  through `start-node.sh` with production's prompt logprobs.
- A fresh download of the 165.3 GiB checkpoint through `./start.sh`,
  including its subdirectory shard, has not been run; the pair had a copy.

## Performance gaps

These come from two different kinds of comparison and should not be read as
one table.

**Same-pair, 2026-10-09.** Mia's TensorFold v1.10 was run on our pair for RC2
([receipts](../results/2026-10-09-rc2/RESULT.md)). On that probe Atlas is
ahead on staggered C4 (~16K: 49.9 s vs 60.2; ~32K: 77.3 s vs 100.1) and on
the Matrix (136.3 s vs 166.9), while TensorFold is faster at single-stream
decode on the same probe prompts (about 49.8 vs 42.2 tok/s prose, 68.5 vs 54.6
code). TensorFold is exl3-based, whose experts are about 8% smaller per rank
than our NVFP4 experts, and our MoE decode kernels already read weights at the
GB10 bandwidth roofline.

**External, published not re-run.** Against the published RiNGSiDE vLLM TP2
figures on their RigMark cell set, Atlas is ahead on single-stream decode,
cold prefill at 8K-64K and newcomer time to first token under staggered
arrivals. Their short-code result is 97.3 tok/s at six streams; the
FP4-Local release image measured 81.4 tok/s at six on the 8 x 128K profile
(90.5 at eight), so at six streams their recipe is ahead.
Older comparisons, including the mmastrac vLLM TP2 figures, remain in their
dated result bundles.

Against kindlingai's published TP=2 vLLM figures (NVFP4 checkpoint, their
`bench.py` and `prefill.py`, run by us on FP4-Local), SparkGLM is ahead on
single-stream decode and level at four mixed streams, but behind at eight
mixed streams (100.6 against their 109.6, remeasured at 116) and on cold 32K
prefill with the shipped 8K chunks (2,690 against 2,931 tok/s; 2,886 with 16K
chunks, which halve our KV pool).

## Behaviour

- The pinned checkpoint template opens reasoning even when a client sends
  `enable_thinking: false`. The candidate follows its default Max effort or
  the template's low/high effort selection. It has no automatic short
  thinking budget. An explicit thinking-token budget can cap reasoning;
  otherwise `max_tokens` can end the request while it is still reasoning,
  before answer content or a tool call appears. Reasoning is returned in
  `reasoning_content`. Budgeting must include both reasoning and the answer.
- RC2 scored 491/500 on the larger arithmetic set and 24/24 on two-hop 24K
  needles. Arithmetic scores across the tested engine configurations ranged
  from 484 to 495; single runs do not establish a quality ranking. Earlier
  small-probe scores remain in their dated result bundles.
- Four greedy prompts run one at a time at C1, then all four together
  at C4, twice, gave the same output every time: each C4 run matched the C1
  output and the two C4 runs matched each other (batched verify, adaptive
  draft width, copy drafts and the prefix cache on;
  `raw/rc2-spark-rc2-f2b805e7.json`, against 0 of 4 on the 2026-10-09 morning
  release). This is the measured four-prompt result, not a general proof.
  Prefill is not row-invariant, so a prompt long enough to prefill in several
  chunks (over 8K tokens) can still differ when the engine is busy (see
  "Not yet measured").
- Long prompts were not reproducible before 2026-09-30. A missing barrier in
  the kernel that normalizes sparse-index keys let one in roughly 60,000
  prompt tokens store a key normalized with a wrong mean. The key stayed in
  the index cache, so the same long prompt could select slightly different
  tokens from run to run and on the two Sparks. Answers in our retrieval
  tests stayed correct. With the fix, four runs of a 50K-token prompt hash
  identically at every traced stage on both ranks.
- Sparse attention uses GLM-5.3's full 2051-candidate selection. The vLLM
  version's 2048-candidate approximation does not apply here.

## Operations

- The API has no authentication and listens only on rank 0's loopback.
- GB10 memory is shared with the host. Running other large workloads on
  either Spark while serving can hang the host.
- Spark uses DFlash2, which is licensed for non-commercial use only.
  Strict uses serial target execution or the checkpoint's native MTP head,
  but the current installer still requires the DFlash2 artifact during
  setup. Selecting Strict does not remove bundled artifact license terms.
- The image is built for arm64 and SM121 (GB10) only.

## Four-Spark TP4 candidate

- The unified RC provides Strict and Spark on TP2 and TP4 in one image.
  `start.sh` manages two nodes; TP4 uses four per-node launcher invocations
  or an orchestrator. `4x1m-tp4` supplies world/TP/EP4, four sequences, a
  1,048,576-token request limit, 8K prefill chunks and memory utilization 0.72.
  These limits do not prove four independent full-window contexts fit.
- Chain is the default. Ring is explicit and requires closing 3–0 links on
  every configured rail. Missing links or mismatched topology fail startup;
  there is no automatic chain fallback. The launcher does not change host
  networking. Ring speedup remains unmeasured for this RC.
- The native-FP8 MLA fusion and native sparse-prefill 16-head path are ported
  to TP4. The latter requires the matching ABI2 library bundled in the image.
  Pair-specific inactive research lanes, including independent/non-speculative
  C4 decode, remain unsupported. Historical K5 flags can also be inactive
  under canonical verification; flag presence alone does not prove execution.
- Strict native MTP is an explicit option, with a full predictor on rank 0 and
  target execution on all four ranks. The focused ring and chain Strict smoke passed
  C1/C4 native-versus-serial token checks, resident-prefix forks,
  cancellation/reuse, JSON/tools and sampled logprobs. The separate near-1M
  capacity check is described above. The TP2-specific native KV cap is not
  copied to TP4.
- TP4 Strict shared-expert splitting is off. A same-image control changing
  only that switch corrected the C4 arithmetic result from `-382130` to
  `-381530` in all four serial and all four native requests; native token IDs
  matched serial. Spark was already unsplit, and TP2 Strict is unchanged.
- Disk cache paths separate rank count, topology and mode. TP4 token-sharded
  KV spill/restore uses the same `--prefix-cache-dir` option. Two cycles
  restored on all four ranks with exact cold/warm native/serial outputs,
  using a 2,048-block diagnostic GPU pool, the normal 48 GiB combined disk
  tier, 16 GPU SSM slots and two host-tier slots.
- The earlier cache mismatch reproduced without eviction: another request
  rewrote a matched 16-token prefix during recomputation. The same-binary
  control passed after enabling only `ATLAS_GLM_PC_WRITE_FLOOR=1`, followed
  by the two-cycle disk check. All four release profiles enable this
  protection; the configuration update leaves the binary/native libraries
  unchanged.
- Sampled verification requires the exchange kernels and protocol on every
  rank. Unsupported sampling/logprob requests fail before verify dispatch.
  A failed exchange after dispatch can terminate the ranks and interrupt
  other requests; peer lifelines must remain enabled.
- The [unified RC result](../results/2026-10-21-unified-rc/RESULT.md) tracks
  remaining Spark/TP2 validation. The native-versus-serial speed comparison
  has one run per workload/arm; it does not measure chain-versus-ring scaling
  or establish broad workload quality.

## Structured output

- `response_format` (`json_schema` or `json_object`) is enforced token by token
  with a grammar, including during speculative decoding. A strict schema the
  engine cannot enforce is refused with HTTP 400 (an error event when
  streaming). It never silently falls back to free text.
- The candidate's sampled speculative verification applies temperature,
  sampling filters, penalties and grammar masks at verify positions on TP2
  and TP4. Profiles also enable masked verification for greedy requests,
  including thinking-boundary masks applied per row. Greedy selection remains
  argmax after processing, and currently gathers full rows on the split head.
  The throughput effect still requires measurement. Grammar or non-neutral penalties
  use full vocabulary rows; requested logprobs also gather complete rows
  before the worker verdict. This corrects the earlier greedy-only split-head
  behavior, but source support does not establish distribution parity or
  quality for every grammar and sampling setting. Final qualification must
  compare uninterrupted serial and speculative requests on the same image.
- A very long strict list at temperature 0 can fall into repetition. Asked for
  60 items, the model produced 121 and degraded into nonsense until
  `max_tokens`. The JSON stays valid, but the content is not usable. Requests
  at our clients' settings (temperature 0.1-0.25, bounded or keyed objects)
  were not affected.
- Every `json_schema` is enforced as strict, including `"strict": false`.

## Installer notes (FP4-Local release)

- The legacy NVFP4 override reuses an existing MTP overlay when the converter is unchanged; `install/convert.sh` does not record the source checkpoint's revision. If you point `CHECKPOINT_REVISION` at a different NVFP4 commit, delete `atlas-overlay` first.
- `serve.py`'s MTP-readiness check reads `quantization_config.quantized_layers` as a mapping; a malformed config makes startup refuse (fail closed), and the engine validates the checkpoint again at load.
