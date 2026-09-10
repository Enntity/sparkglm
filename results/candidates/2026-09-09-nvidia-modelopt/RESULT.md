# NVIDIA ModelOpt NVFP4 integration

The optional `nvfp4-nvidia` profile loads and serves the pinned NVIDIA model on
TP2 DGX Spark hardware with native FP4 dense/MoE kernels and MXFP8 DFlash2.
It remains an experiment. Red Hat remains the default. No speed, quality,
512K-capacity, or release promotion follows from this bounded integration run.

## Verified integration

- Both ranks independently verified all 33 weight shards against the published
  Hugging Face SHA-256 hashes for revision
  `423acf37583782c51c142d145aef733d72943d93`.
- G0 source checks and publication audit passed.
- G1 dense numerical checks at TP2 shapes passed below 0.03 relative L2,
  including changed-input CUDA graph replay. The existing clamped MoE
  numerical/profiler probe passed with native SM120 FP4 kernel evidence.
- G2 three repetitions each of C1 decode, mixed C4 and long C2 completed
  deterministically on a synthetic dense-KDA/routed-KDA/sparse-MLA fixture.
- Full-model load, target/draft graph capture and an authenticated gateway
  completion succeeded. Logs identify FlashInferCutlassNvFp4LinearKernel and
  FLASHINFER_CUTLASS NvFp4 MoE. Reported model allocation was 89.65 GiB on
  rank 0; this is not total resident/system memory. Graph capture added 0.77 GiB.
- All five retained streaming workloads delivered the requested 400 tokens per
  request, included their own marker, and had no foreign request markers or
  request errors.

## Bounded comparison, not G3 qualification

Same serving image pair, TP2, draft TP2, DFlash2 MXFP8 revision
`610aa967a92bfeb97e3d848dcb8693553e8b6a55`, seven draft tokens, FP8 KV,
9 GiB KV per rank, configured 524288 context, 2048 prefill chunk and four
sequences. The target checkpoint/quantization loader is the intentional change.
Inference used the authenticated gateway; exact token counts used the private
backend tokenizer. Corresponding request prompt hashes match across arms.
One exact C4 warmup was discarded for each arm. All transfers/checksum scans
finished before the retained runs. Run order was Red Hat then NVIDIA, one
retained repetition each; this does not meet the three alternating pairs
required for G3 or support a general speed claim.

| Workload | Red Hat wall seconds | NVIDIA wall seconds |
| --- | ---: | ---: |
| C1-16K | 20.41 | 22.79 |
| C1-32K | 26.59 | 29.12 |
| C2-16K | 30.88 | 33.52 |
| C2-32K | 55.19 | 50.74 |
| C4-16K | 61.21 | 62.53 |

The observed pattern is mixed. Full per-request TTFT, visible SSE gap and
completion receipts are under `raw/`. API usage reported zero prediction-token
fields; that is not evidence of no speculation. Actual vLLM lifetime counters
recorded 4372 accepted tokens out of 9565 drafted (45.7%). These counters cover
the canary, semantics, warmup and retained matrix together, not a paired
acceptance benchmark and not an MTP result.

## Quality and operational limits

The same 16-case semantic suite scored 15/16 for Red Hat and 14/16 for NVIDIA.
Both passed plain output, reasoning, structured JSON, parallel tools and tool
results, prefix miss/hit/isolation, stop, vision, stream cancellation and the
post-cancellation request. With thinking disabled, NVIDIA answered `123*47`
as `5631` (correct: `5781`) and `91*89` as `8119` (correct: `8099`). Red Hat
also missed `91*89`, answering `8199`. This bounded suite fails the quality
gate; it is not a general model-quality ranking. No endurance, cold-start
repetition, maximum-context stress or partial-node-failure qualification ran.

The installed images predate this profile-only change; their exact digests
are recorded in the server manifests. The source revision identifies the
profile/harness, not a claim that these existing images were rebuilt from it.
CPU-host download staging and worker runtime registration were corrected
before full-model startup. No serving-engine patch was needed for DFlash.

## MTP

MTP is worth a separate experiment, but is not enabled by this integration.
The pinned checkpoint's MTP layer contains 14,865,185,408 tensor bytes:
888 BF16 tensors and one FP32 tensor. The installed vLLM MTP configuration
inherits `modelopt_fp4`; its prediction block passes that quantization to the
MLP. The checkpoint ignore list stops at layer 44. The checked-in preflight
confirms representative BF16 layer-45 expert/shared-expert weights are not
excluded from FP4. The FP4 loader expects packed bytes and quantization scales.
A separate BF16-compatible loader/backend path must pass numerical and tiny
integration checks before MTP generation and acceptance/speed comparisons.
This is a metadata/source compatibility finding, not a failed MTP generation
run or a performance rejection. No weight values were modified or published.

## Operational outcome

NVIDIA was stopped after the test and remains installed as a separate on-demand
runtime. The prior Qwen runtime and Presence local-first route were restored;
strict-local and Presence gateway canaries both returned `READY` with HTTP 200.
The temporary synthetic model registration was removed from both ranks.
No default, public main branch, or release changed.
