<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# The engine

SparkGLM runs [Atlas](https://github.com/Atlas-Inf/atlas), built from the
public fork [`Enntity/atlas`](https://github.com/Enntity/atlas) at the commit
pinned in [`install/atlas-source.json`](../install/atlas-source.json), for the
kernel target `glm-5.3-flash`. [`install/build.sh`](../install/build.sh)
refuses any other tree.

Both Spark products use the common `spark-shared` engine. This candidate pins
`adb7a5b381ce0f8cafc29e37d11622c63e6ebdb9` (tree `a2965af0d0e9de275523855b7bcf247c0bc0b944`).
Model-specific execution remains selected by kernel target and checkpoint type.
The shared media decoder retains GLM's reference frame sampler, canvas sizing,
normalization and final-frame padding alongside Qwen's timestamped video path.
Qualification applies to the exact product image, not the branch name alone.
GLM and Qwen preserve literal XML examples inside streamed reasoning. GLM's
native tool opener still ends reasoning and enters the validated tool parser;
legacy model leak guards remain in place.

The unified TP2/TP4 release provides Strict and Spark precision policies, corrected
FP4 weight rounding, native MTP with prefix caching, native-FP8 MLA fusion,
and sparse attention for TP2's 32 or TP4's 16 local heads. The same image runs
an explicit four-rank chain or ring. The install manifest selects the exact
source. [Release evidence](../results/2026-10-21-unified-release/RESULT.md)
records the completed checks and the scope of reused TP2 results.

The earlier release was `sparkglm/atlas-20261017-fp4local` at `b3b016ab`
(tree `14316380`), built on the shared-engine release
([2026-10-15 results](../results/2026-10-15-shared-engine/RESULT.md)).
Its [2026-10-17 results](../results/2026-10-17-fp4local/RESULT.md) remain
historical measurements. The README describes the [engine layers](../README.md#the-engine).

The release also fixes speculative returned scores under logit bias and
history penalties. Requested scores use the serial processed distribution;
unsupported speculative scoring paths use a supported verification or serial
path. First-prefill-token logit-bias support remains an inherited limitation.

Accepted prediction usage counts only draft tokens retained in the response,
including token-limit, stop and rollback cases. Internal verification and
draft-depth statistics retain their existing meaning.

## Serving behavior

The launcher resolves the mode over a capacity profile; the profile's raw
`environment` alone is not the effective serving policy. Every rank must
resolve the same mode and topology.

- **Strict** preserves native weights and checkpoint input scales and uses
  A16 routed decode. It runs the canonical serial target by default. Native
  checkpoint MTP is opt-in with `SPARKGLM_STRICT_MTP=1`, at C1..C4 with prefix
  caching and the profile's context limit. The full predictor runs only on
  rank 0; every rank runs target verification and prefix/admission agreement.
- **Spark** uses calibrated A4 routed decode, DFlash2 and corrected
  round-to-nearest-even FP4 KDA decode/shared-expert weights. KDA prefill,
  MLA, dense projections and the vocabulary head remain native. Shared-expert
  TP splitting is off: it introduced additional partial-output rounding in
  the arithmetic regression. Drafter weight conversions also use corrected
  rounding; activation quantizers are not changed by that rounding fix.

The TP4 profile disables shared-expert TP splitting in Strict as well. In a
same-image C4 control, changing only `ATLAS_GLM_SHARED_TP_SPLIT` from `1` to
`0` corrected all four serial and four native-MTP arithmetic answers from
`-382130` to `-381530`; native token IDs matched serial. TP2 Strict's setting
is unchanged. This control establishes the TP4 policy for that regression,
not a performance gain or general numerical equivalence.

Both modes reject missing required formats/calibration instead of silently
falling back to another weight conversion. Neither removes FP8 KV, dynamic
prefill activation quantization, floating-point rounding, or chunk-scheduling
effects. Strict describes the precision policy, not a cross-engine bitwise
identity guarantee.

TP4 uses the native-FP8 MLA fusion and the sparse-prefill library's 16-head
path. The library has ABI2 with an explicit 16/32-head geometry, and is rebuilt
with the engine; old ABI1 libraries are rejected before serving. Existing
sequence-parallel prefill, sparse decode and staged projections already handle
TP4. Some historical K5 switches remain present in profiles but are inactive
under canonical verification, including `ATLAS_GLM_K5_FUSED_TP_HC`; copying a
switch value is not evidence that its path executes.

TP4 chain and ring use the same arithmetic and rank placement. Ring requires
explicit left/right links on every rank, including rank 3–rank 0. Its performance
must be measured on the resulting image and fabric; no speedup is assumed.

TP4 sharded KV now supports the NVMe prefix tier using the same
`--prefix-cache-dir` launcher option. Cache namespaces still separate TP2
from TP4, Strict from Spark, and the TP4 chain from ring.

The release profiles enable `ATLAS_GLM_PC_WRITE_FLOOR=1`. A control without
eviction reproduced a cache mismatch after another request matched a
16-token prefix without a usable recurrent-state snapshot, then recomputed
and rewrote those shared KV rows. Changing only this switch on the same
binary preserved the matched blocks and restored exact cold/warm
native-versus-serial outputs. Two subsequent NVMe eviction/restore cycles
passed on all four ranks. The correction changes profile configuration;
the engine executable and native libraries are unchanged.

Strict smoke passed on both TP4 ring and chain: C1/C4 native-versus-serial
token checks, resident-prefix forks, cancellation/reuse, constrained JSON,
tools and sampled full-vocabulary verification with finite logprobs. The
ring also passed the separate 1,046,988-token shared-prefix capacity check,
including four distinct resident branches at decode width four. The disk
test used a 2,048-block diagnostic GPU pool, the normal 48 GiB combined disk
tier, 16 GPU SSM slots and two host-tier slots. These checks and the focused
native/serial speed comparison have separate
[receipts and scope](../results/2026-10-21-unified-rc/RESULT.md).

The checkpoint chat template opens a reasoning block, including when a
client sends `enable_thinking: false`. Its default is `Reasoning Effort: Max`;
the template recognizes low and high effort. The release removes the
old automatic short reasoning cap. An explicit client thinking-token budget
can cap reasoning; otherwise it ends when the model closes the block or
the request reaches `max_tokens`. That limit includes reasoning, so a short
request can end before producing an answer. Sampling defaults come from
`generation_config.json` (temperature 1.0, top_p 0.95 for the pinned
checkpoint), with top_k, min_p and top_n_sigma off unless requested.

With the supplied profiles and default sampling policy, TP2 and TP4
speculative verify positions use the request's sampling filters, penalties
and grammar masks. A sampled target token accepts its draft only while the
tokens match; the first mismatch ends that accepted prefix. The split
vocabulary head first exchanges candidates when they suffice. Grammar,
non-neutral penalties, undecided candidate sets and requested logprobs use
full vocabulary rows. Every profile enables `ATLAS_DFLASH_MASKED_VERIFY=1`.
On the split head, unstructured greedy verification currently gathers full
vocabulary rows and applies the same per-row thinking-boundary masks as the
serial pipeline. Greedy selection uses the processed row's argmax. This may
cost throughput; performance claims require matched measurements.

Split and scored verify spans preserve their per-row boundary decisions when
committing an accepted prefix. They do not replay the raw forced-end helper
against the span's earlier fence state. The generic legacy fallback path is
unchanged; the source fix does not establish equivalence for every model or
configuration.

A split-head request needing sampling or logprobs is rejected before verify
dispatch if the required exchange protocol is unavailable. An exchange
failure after dispatch terminates the affected EP rank because its peers may
be waiting for a verdict; enabled peer lifelines shut down the other ranks.
This can interrupt other requests. The implementation does not substitute
greedy picks after such a failure. Distribution and quality qualification
must still use the exact built image at each topology.

For diagnostics, `/v1/completions` accepts `disable_mtp: true` to run a
request without speculative decoding; draft weights remain resident.
Blocking completions can also request `return_token_ids: true`, which adds
exact generated IDs to `choices[].token_ids` before text stripping and
excludes an echoed prompt. Combining that option with streaming is rejected.
Both options default to false.

## FP4-Local series

The 2026-10-17 series comprised 112 commits on `16315004`. Commits marked `shared:` in their subject are
model-independent and are candidates for the shared layer; until it moves they
sit in the `sparkglm` layer.

- **Loading in the 2026-10-17 release.** NVIDIA's `csf_32x2_v1` compressed block scales were decoded
  losslessly at upload, and under expert TP per rank slice. FP4-Local's MXFP8
  and FP8_PB_WO projections were fed to the arm production ran on the NVFP4
  checkpoint, from an exact (MXFP8) or nearest (FP8 block) BF16 copy, and then
  served in that release's formats: KDA `q/k/v/o`, shared experts and dense MLP
  in NVFP4, MLA in MXFP8. The quantized sources are released once a layer is
  built. The MTP experts and the vision MLP load from their W4A16 NVFP4
  weights. The NVFP4 checkpoint comparison used the shared-engine paths
  and gave the same prompt logprobs. These are historical results.
- **Tools with `response_format`.** Under `tool_choice: auto`, a request that
  sends tools and a `response_format` schema gets one tool-call-or-JSON union
  grammar. The model may open GLM's native tool call or answer in the schema,
  and the union picks its branch by probability mass (ToolEvalBench 88.0 to
  91.0).

## Switches the profiles turn on

The shipped profiles ([`install/profiles/`](../install/profiles/)) list every
option they set; `install/test_install.py` checks those settings. The following
prefill options were introduced in the 2026-10-17 release. Each gave the same
48,875-token prompt-logprob hash, `25bc828eea3f`, in that release's comparison:

| Switch | What it does |
|---|---|
| `ATLAS_GLM_KDA_SIDE_PROJ_LT=1` | KDA `beta \| f_a \| g_a` prefill projections as one packed CUTLASS GEMM |
| `ATLAS_GLM_KDA_CONV_PLANES=1` | KDA prefill convolves the q/k/v planes without the packed copy |
| `ATLAS_GLM_HC_PREFILL_FUSE2=1` | The mHC prefill pre-site finalizer also writes the block's input norm |
| `ATLAS_GLM_SP_GATHER_OVERLAP=1` | Sequence-parallel prefill seams overlap their exchanges with their compute |

Historical cold prefill, 2026-10-17, with kindlingai's `prefill.py` (random words, 8K chunks, `4x512k`
at 0.91; [raw/prefill](../results/2026-10-17-fp4local/raw/prefill/)): 2,581 /
2,674 tok/s at 32K / 128K without them, 2,645 / 2,733 with the first three,
and 2,690 at 32K with all four (2,622 with three on that start).

The 2026-10-17 native-format opt-in comparison used W8A16 prefill
(`ATLAS_GLM_FP4L_FP8_PREFILL=w8a16`), the `4x512k` profile and six runs per
workload. It measured decode 10–16% slower: code / prose / structured 72.6 / 39.7 / 94.6 against
82.8 / 47.2 / 104.7 tok/s. ToolEvalBench was 91 for both arms, with about
2.6 GB less memory per rank before the KV pool (+26% blocks). Those receipts
are in the [dated result bundle](../results/2026-10-17-fp4local/RESULT.md).
Current formats are selected by mode; those old timings do not measure
either mode of the unified release.

## Opt-in switches (default off)

None of these is set by a shipped profile. To try one, copy a profile, add the
switch to its `environment`, and pass the file to `start-node.sh --profile` on
every rank (the launcher ignores ambient `ATLAS_*` variables). These historical
experiments are not TP4 qualification.

| Switch | What it does | Measured (receipts in [results/2026-10-17-fp4local](../results/2026-10-17-fp4local/RESULT.md)) | Why it is off |
|---|---|---|---|
| `ATLAS_DFLASH_WARM_CTX=exact\|seed\|restore` | The drafter's 2,048-row prompt window: `exact` anchors every pass at the prompt end, `seed` also starts a prefix-cache restore's window at the first computed row instead of zero rows, `restore` brings the rows under the cached prefix back from a store. `legacy` (the default) is the old placement, byte for byte | Warm decode 77.5 against 76.1 tok/s (`legacy`), one start each; same greedy text | Within run-to-run spread |
| `ATLAS_GLM_PACKED_PREFILL=1` | Packs a scheduler tick's new short prompts into one prefill pass | Mixed 4 / 8 streams 86.4 / 105.7 tok/s against 85.2 / 100.9; median time to first token 1.24 / 1.79 s against 2.44 / 4.67 s (with the two below) | Experimental, not yet exact against unpacked prefill |
| `ATLAS_TAIL_SPLIT_MIN_TOKENS=N` | A prompt under N tokens prefills in one pass instead of two (the tail-checkpoint split); its next turn loses only the tail checkpoint | With `ATLAS_PREFILL_BURST=1`, N = 512: mixed 84.7 / 104.4 tok/s, 1.95 / 3.66 s | Untuned; the gain is in time to first token under load |
| `ATLAS_PREFILL_BURST=1` | A tick that admitted requests and finds more short prompts waiting skips its decode and drains them, so a burst prefills back to back (bounded by `ATLAS_PREFILL_BURST_MAX_MS` and `_MAX_TOKENS`; shared layer) | See above | Untuned |
| `ATLAS_TTFT_TRACE=1` | Logs a per-request time-to-first-token breakdown, from arrival to the first decode step | Diagnostic | Log volume |

Prefill chunk size is a server argument, not a switch. In the 2026-10-17
comparison,
`--max-prefill-tokens=16384` raised cold prefill to 2,886 / 3,002 tok/s at
32K / 128K with the four switches, but took about 4.1 GB more before the KV
pool (77,348 blocks instead of about 166,000) and gives other prompt logprobs
(`46cdaad4c42f`), since prefill is not yet chunk-invariant. The profiles keep
8K.
