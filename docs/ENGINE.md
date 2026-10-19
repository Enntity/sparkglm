<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# The engine

SparkGLM runs [Atlas](https://github.com/Atlas-Inf/atlas), built from the
public fork [`Enntity/atlas`](https://github.com/Enntity/atlas) at the commit
pinned in [`install/atlas-source.json`](../install/atlas-source.json), for the
kernel target `glm-5.3-flash`. [`install/build.sh`](../install/build.sh)
refuses any other tree.

The 2026-10-19 qualification candidate adds the four-rank RDMA chain,
checkpoint-format defaults, checkpoint chat and sampling behavior, FP32
router logits, and sampled speculative verification. This description was
checked against engine source `b6186540c`; the install manifest selects the
exact source used by a build. Source support alone does not qualify an image.
Final TP2 and TP4 performance and quality claims require that image's receipts.

The earlier release was `sparkglm/atlas-20261017-fp4local` at `b3b016ab`
(tree `14316380`), built on the shared-engine release
([2026-10-15 results](../results/2026-10-15-shared-engine/RESULT.md)).
Its [2026-10-17 results](../results/2026-10-17-fp4local/RESULT.md) remain
historical measurements. The README describes the [engine layers](../README.md#the-engine).

## Candidate behavior

Every TP2 and TP4 product profile enables `ATLAS_GLM_FP4L_NATIVE=1`.
KDA projections and the dense MLP use the checkpoint's MXFP8 weights;
MLA and shared experts use its FP8 block weights. This avoids the earlier
NVFP4 requantization of those projections. The profiles also enable
`ATLAS_GLM_FP4L_INPUT_SCALES=1` for calibrated routed-expert prompt
activations and `ATLAS_GLM_FP4L_DECODE_A16=1` for BF16 activations in routed
decode and verify rows. Native weight formats still have their prescribed
activation quantization and floating-point rounding; they do not establish
bitwise agreement with another engine.

The checkpoint chat template opens a reasoning block, including when a
client sends `enable_thinking: false`. Its default is `Reasoning Effort: Max`;
the template recognizes low and high effort. The candidate removes the
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
full vocabulary rows. Greedy requests retain the greedy fast path.

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
Native formats are now enabled by the product profiles; those old timings
do not measure the integrated candidate.

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
