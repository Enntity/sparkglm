<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# The engine

SparkGLM runs [Atlas](https://github.com/Atlas-Inf/atlas), built from the
public fork [`Enntity/atlas`](https://github.com/Enntity/atlas) at the commit
pinned in [`install/atlas-source.json`](../install/atlas-source.json), for the
kernel target `glm-5.3-flash`. [`install/build.sh`](../install/build.sh)
refuses any other tree.

The pin is the release tag `sparkglm/atlas-20261017-fp4local` at `b3b016ab`
(tree `14316380`): the shared-engine release (`sparkglm/atlas-20261015-vision`,
`16315004`; [results/2026-10-15-shared-engine](../results/2026-10-15-shared-engine/RESULT.md))
plus the [FP4-Local series](#fp4-local-series)
([results/2026-10-17-fp4local](../results/2026-10-17-fp4local/RESULT.md)).
The layers are described in the README's [engine section](../README.md#the-engine).

## FP4-Local series

112 commits on `16315004`. Commits marked `shared:` in their subject are
model-independent and are candidates for the shared layer; until it moves they
sit in the `sparkglm` layer.

- **Loading.** NVIDIA's `csf_32x2_v1` compressed block scales are decoded
  losslessly at upload, and under expert TP per rank slice. FP4-Local's MXFP8
  and FP8_PB_WO projections are fed to the arm production runs on the NVFP4
  checkpoint, from an exact (MXFP8) or nearest (FP8 block) BF16 copy, and then
  served in production's formats: KDA `q/k/v/o`, shared experts and dense MLP
  in NVFP4, MLA in MXFP8. The quantized sources are released once a layer is
  built. The MTP experts and the vision MLP load from their W4A16 NVFP4
  weights. With the NVFP4 checkpoint every path is the one the shared-engine
  release ran (same prompt logprobs).
- **Tools with `response_format`.** Under `tool_choice: auto`, a request that
  sends tools and a `response_format` schema gets one tool-call-or-JSON union
  grammar. The model may open GLM's native tool call or answer in the schema,
  and the union picks its branch by probability mass (ToolEvalBench 88.0 to
  91.0).

## Switches the profiles turn on

The shipped profiles ([`install/profiles/`](../install/profiles/)) list every
option they set; `install/test_install.py` checks that the exact ones are on and
the held-back ones are absent. New in this release, all prefill-only and
exact (the same 48,875-token prompt-logprob hash, `25bc828eea3f`, with each on):

| Switch | What it does |
|---|---|
| `ATLAS_GLM_KDA_SIDE_PROJ_LT=1` | KDA `beta \| f_a \| g_a` prefill projections as one packed CUTLASS GEMM |
| `ATLAS_GLM_KDA_CONV_PLANES=1` | KDA prefill convolves the q/k/v planes without the packed copy |
| `ATLAS_GLM_HC_PREFILL_FUSE2=1` | The mHC prefill pre-site finalizer also writes the block's input norm |
| `ATLAS_GLM_SP_GATHER_OVERLAP=1` | Sequence-parallel prefill seams overlap their exchanges with their compute |

Cold prefill with kindlingai's `prefill.py` (random words, 8K chunks, `4x512k`
at 0.91; [raw/prefill](../results/2026-10-17-fp4local/raw/prefill/)): 2,581 /
2,674 tok/s at 32K / 128K without them, 2,645 / 2,733 with the first three,
and 2,690 at 32K with all four (2,622 with three on that start).

## Opt-in switches (default off)

None of these is set by a shipped profile. To try one, copy a profile, add the
switch to its `environment`, and pass the file to `start-node.sh --profile` on
both ranks (the launcher ignores ambient `ATLAS_*` variables).

| Switch | What it does | Measured (receipts in [results/2026-10-17-fp4local](../results/2026-10-17-fp4local/RESULT.md)) | Why it is off |
|---|---|---|---|
| `ATLAS_GLM_FP4L_NATIVE=1` | Serves every group in FP4-Local's own format, exactly: KDA `q/k/v/o` and the dense MLP on their MXFP8 bytes, MLA and the shared experts on their FP8 block bytes. Per-group switches (`ATLAS_GLM_FP4L_KDA_MXFP8_DECODE`, `ATLAS_GLM_FP4L_MLA_FP8`, `ATLAS_GLM_FP4L_SHARED_FP8`, `ATLAS_GLM_FP4L_DENSE_MXFP8`) override it one group at a time; `ATLAS_GLM_FP4L_FP8_PREFILL` picks the native prefill (`w8a8`, the default, `w8a16` or `bf16`) | Decode 10–16% slower (code / prose / structured 72.6 / 39.7 / 94.6 against 82.8 / 47.2 / 104.7 tok/s); ToolEvalBench 91, the same; about 2.6 GB less per rank before the KV pool (+26% blocks). The KDA GEMV already runs at the copy floor of its bytes | On in every profile since 2026-10-19: the NVFP4 requant lost long-context retrieval (two-hop needle 21 vs 32 of 60) |
| `ATLAS_DFLASH_WARM_CTX=exact\|seed\|restore` | The drafter's 2,048-row prompt window: `exact` anchors every pass at the prompt end, `seed` also starts a prefix-cache restore's window at the first computed row instead of zero rows, `restore` brings the rows under the cached prefix back from a store. `legacy` (the default) is the old placement, byte for byte | Warm decode 77.5 against 76.1 tok/s (`legacy`), one start each; same greedy text | Within run-to-run spread |
| `ATLAS_GLM_PACKED_PREFILL=1` | Packs a scheduler tick's new short prompts into one prefill pass | Mixed 4 / 8 streams 86.4 / 105.7 tok/s against 85.2 / 100.9; median time to first token 1.24 / 1.79 s against 2.44 / 4.67 s (with the two below) | Experimental, not yet exact against unpacked prefill |
| `ATLAS_TAIL_SPLIT_MIN_TOKENS=N` | A prompt under N tokens prefills in one pass instead of two (the tail-checkpoint split); its next turn loses only the tail checkpoint | With `ATLAS_PREFILL_BURST=1`, N = 512: mixed 84.7 / 104.4 tok/s, 1.95 / 3.66 s | Untuned; the gain is in time to first token under load |
| `ATLAS_PREFILL_BURST=1` | A tick that admitted requests and finds more short prompts waiting skips its decode and drains them, so a burst prefills back to back (bounded by `ATLAS_PREFILL_BURST_MAX_MS` and `_MAX_TOKENS`; shared layer) | See above | Untuned |
| `ATLAS_TTFT_TRACE=1` | Logs a per-request time-to-first-token breakdown, from arrival to the first decode step | Diagnostic | Log volume |

Prefill chunk size is a server argument, not a switch:
`--max-prefill-tokens=16384` raised cold prefill to 2,886 / 3,002 tok/s at
32K / 128K with the four switches, but took about 4.1 GB more before the KV
pool (77,348 blocks instead of about 166,000) and gives other prompt logprobs
(`46cdaad4c42f`), since prefill is not yet chunk-invariant. The profiles keep
8K.
