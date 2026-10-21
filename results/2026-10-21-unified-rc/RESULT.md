# SparkGLM unified TP2 / TP4 RC — 2026-10-21

One image now supports Strict and Spark on TP2 and TP4, with explicit switchless chain or ring on TP4. The scoped release checks pass. This is a local RC package, not a public release or a claim that a new full benchmark campaign was run.

## Package and usage

- Image tag: `localhost/atlas-sparkglm:unified-rc-46f14bee214a`. The verified image is present on all four Sparks.
- Engine commit: `d3620ffbaf654186c7213b320e52d0112a71f956`.
- Install tree: `46f14bee214afe112fa6199e719a60586da00ffc`.
- Binary SHA-256: `e79d163dd10c07e3244de7c26a73c915a35ae3df08d5db34162243baaf182467`.
- All-host image content projection SHA-256: `2dcaaa507e4acdea7807cfdebc3e1dc7d9efef28a0b59e251c0d7198a4c7214b`. Docker classic/containerd image IDs differ between pairs; full Config and RootFS content match.
- `build.sh` in the source bundle uses the complete pinned source snapshots and the product Dockerfile. It requires Linux arm64, Docker/BuildKit, network access for build dependencies, and separately supplied model checkpoints. The verified incremental image was built and tested; the portable cold-build recipe was syntax-checked, not redundantly rebuilt.

Use the existing `install/start-node.sh` entry point on every rank: `--mode strict` or `--mode spark`; Strict native MTP adds `--strict-mtp`. TP4 adds `--world-size 4 --topology chain|ring` with explicit left/right rails. Chain omits the endpoint's absent neighbor; ring requires both rails across the 01–04 closing edge. The launcher rejects incomplete or inconsistent topology instead of silently falling back. See the bundled README and manual for complete commands.

Strict preserves checkpoint weight formats and input scales with A16 routed decode. Spark retains corrected weight-RNE FP4 KDA decode and shared experts, calibrated A4 routed decode, DFlash2, native prefill/MLA, tiled layouts, fusions and communication optimizations. Shared-expert TP splitting stays off in Spark and is also off in TP4 Strict after a matched arithmetic failure. TP2 Strict's prior policy is retained. Existing FP8 KV remains in both modes; Strict is a precision policy, not a cross-engine bitwise promise.

## Focused final-image validation

| Arm | Checks | Requests | Result |
|---|---:|---:|---|
| tp4-chain-spark-final | 15 | 22 | pass |
| tp4-ring-spark-final | 15 | 22 | pass |
| tp4-ring-strict-native-protected | 19 | 31 | pass |
| tp2-chain-strict-native-final | 19 | 31 | pass |
| tp2-chain-spark-final | 15 | 22 | pass |

These checks cover C4 arithmetic (`-381530`), JSON and tool use, concurrent prefix reuse, cancellation and slot reuse. Strict checks native-versus-serial token IDs and sampled full-vocabulary logprobs. Spark also checks code/structured tasks and the original 70,706- and 113,161-token retrieval cases. Both long retrieval answers pass on each tested Spark arm. Clocks were uncapped during measurements; per-rank startup/configuration/clock receipts are retained.

## Chain versus ring — Spark TP4

Same final image, requests, sampling settings and server argv; only topology, neighbor rails and topology-specific cache namespaces differ. Every output, reasoning and full-stream hash matches its paired repetition. All requests finish naturally below the 4,096-token output budget.

| Workload | Chain median tok/s | Ring median tok/s | Ring change |
|---|---:|---:|---:|
| prose | 57.145 | 58.328 | +2.07% |
| code | 104.147 | 106.368 | +2.13% |
| structured | 137.936 | 141.363 | +2.48% |

Every repetition:

| Workload | Run | Chain tok/s | Ring tok/s | Completion tokens | Output parity |
|---|---:|---:|---:|---:|---|
| prose | 1 | 58.400 | 59.636 | 971 | exact |
| prose | 2 | 55.889 | 57.019 | 1008 | exact |
| code | 1 | 105.312 | 107.523 | 1783 | exact |
| code | 2 | 102.981 | 105.214 | 2080 | exact |
| structured | 1 | 135.843 | 139.470 | 446 | exact |
| structured | 2 | 140.029 | 143.256 | 441 | exact |

Two repetitions per case are a quick comparison, not a significance study. Ring shortens the small-collective flood and gather path; specialized reduce-scatter, all-to-all and command forwarding retain their established linear schedules and arithmetic.

## Native MTP and near-limit capacity

Strict native TP4 processed a 1,046,988-token prompt and matched serial on the exact retrieval marker. Four identical cached branches and four distinct-tail branches matched their controls. Both bursts reached decode width four; distinct branches overlapped as resident sequences for 8.539699 seconds. Cancellation and slot reuse passed. Minimum sampled host MemAvailable was 25.209 / 45.570 / 39.974 / 45.678 GiB on 01 / 02 / 03 / 04. These are four branches sharing one near-1M prefix, not four independent million-token contexts.

On the same engine before the final write-protection profile setting, one native-versus-serial speed sample per workload measured prose 41.531 vs 39.888 tok/s, code 53.066 vs 39.586, and structured 57.021 vs 39.991. Output, reasoning and stream hashes matched. This is explicitly carried candidate03 evidence, not a new performance measurement of the final cache policy. The final protected-policy Strict smoke covers the affected prefix/native path. The final profile does not change memory allocation or context limits.

## Two correctness fixes found during qualification

The first ring mixed-collective test exposed a cross-QP completion race: a newer completion flag could overtake an older payload or be overwritten by an older completion. Completion words now follow the payload's parity slot; arithmetic and TP2's single-word protocol are unchanged. The CPU regression deliberately delays the competing QP and passes 63,979,520 exact BF16 comparisons. All eight GPU chain/ring transport cases pass, including mixed collectives, all-to-all, fused and FP8 reduce-scatter. The wire-version change refuses mixed old/new TP4 builds.

The forced NVMe test initially changed the answer from 50 to 54 streamed token IDs. Cold and resident-warm controls matched, but a further control reproduced the same change after one pressure request with **no disk eviction or restore**. That request recomputed a shared 16-token prefix and rewrote blocks still used by the cached target and predictor. Changing only `ATLAS_GLM_PC_WRITE_FLOOR` to `1` prevents those writes. All four release profiles now enable it with subblock caching off and incompatible legacy/exact-end policies absent. Engine and native-library bytes are unchanged by this fix.

With protection on, the original prompt and pressure sequence retain exact cold/warm and native/serial IDs through two forced disk restores: each rank restores 160/160 then 156/156 blocks with zero failures. The probe uses a diagnostic 2,048-block target pool to force eviction quickly, with the normal 48 GiB combined disk tier (24 GiB KV + 24 GiB SSM per rank), 16 GPU snapshot slots and two host-tier slots. The default serving profile does not impose that small pool. Failed controls and the causal comparison remain in the archive; no four-class storage defect is claimed.

## Evidence and scope

The TP4 native sparse-attention port supports 16 and 32 local heads; the exact GPU oracle compared 676,282,368 output bytes and 2,641,728 LSE bytes. Native MLA fusion passed actual TP2/TP4 shape tests. The final engine build passed 65 runtime NVMe tests and 39 model storage tests; the real CUDA/NVMe test covered eight rank/path cases and 401,408 exact bytes. Two intentionally ignored runtime tests were the existing benchmark and GPU-only case; the GPU case was then run separately with the real driver. The profile-only image passed 56 Linux launcher tests, with converter/shell/syntax checks and engine formatting verified separately.

Earlier TP2 full qualification and the surgical arithmetic/native-MTP fixes are preserved under `prior-tp2/`. This work reruns the affected TP2 checks and newly implemented TP4 functionality. It does not repeat Kindling, RigMark, NLL, the full needle suite or a sampler campaign. `qualification.json` records exactly which evidence was reused and why. Raw requests, outputs, logs, image attestations, source identities, component tests, failed controls and all repetitions are in `receipts.tar.gz`; `SHA256SUMS` covers the deliverables.

Managed Qwen on 03/04 and the prior managed GLM on 01/02 were restored through LLooM, with original-image checks, normal clock caps and real gateway answers of `42`. The girls' Presence route was returned from throttled DeepSeek to Qwen and verified. Each restoration receipt records its verification time, before subsequent work by other owners. This RC was not pushed or publicly published.
