# Sparse physical-index reuse rejection — 2026-09-02

Candidate commit `c23cbc0` ports vLLM PR #49678 at source commit
`5a7de279e309adfb212681d5f691961ed079c8e5` to the active
`FLASHINFER_MLA_SPARSE_SM120` path. The candidate image was
`sparkglm-vllm:physical-topk-reuse`; the unchanged control was
`sparkglm-vllm:d3860af`.

## Verdict

Reject. The candidate improved measured prefill and TTFT, but corrupted every
output in the frozen medium/large C1/C2 matrix. Commit `0b005440` removes the
candidate from the appliance branch. It remains preserved at
`experiments/rejected-physical-topk-reuse` for diagnosis.

The matrix runner now fails closed on missing or foreign isolation markers, so
this class of fast-but-wrong result cannot pass the workload gate again.

## Same-shape A/B

Both runs used the same two-Spark TP2 recipe, 2,048-token scheduling budget,
five-second C2 stagger, 64 forced output tokens, and cold per-image prefix
caches. Candidate values are compared with the fresh `d3860af` control run.

| Case | Candidate TTFT (s) | Control TTFT (s) | Candidate prefill (tok/s) | Control prefill (tok/s) | Candidate/control wall (s) |
| --- | --- | --- | --- | --- | --- |
| Medium C1 | 12.51 | 15.62 | 1264.0 | 1011.9 | 17.71 / 17.92 |
| Medium C2 | 14.60 / 26.28 | 17.96 / 29.01 | 1082.5 / 601.4 | 880.1 / 544.9 | 36.87 / 36.19 |
| Large C1 | 26.83 | 31.73 | 1199.5 | 1014.1 | 32.42 / 34.03 |
| Large C2 | 28.42 / 54.83 | 35.53 / 61.18 | 1132.2 / 586.9 | 905.7 / 525.9 | 65.55 / 68.27 |

The apparent gains were real timings but invalid optimization results. All six
candidate outputs collapsed to the same SHA-256 and failed their own isolation
marker. A focused candidate probe returned the repeated text
`lockhandlehandlehandle...`, had `own_request_marker=false`, and stopped only at
the forced 64-token limit.

The exact same salted focused probe against `d3860af` returned a coherent
summary beginning with its `GLMISO-9891E84D8A3A0A849FC0` marker, no foreign
marker, and all 64 requested tokens. Its TTFT was 16.16 seconds, effective
prefill was 977.5 tok/s, and decode was 26.9 tok/s. This isolates the regression
to the physical-index reuse candidate rather than the packed-RMSNorm control.

## Failure boundary

The upstream conversion allocates an output initialized to `-1` because its
compaction kernel leaves unused row tails unwritten. The port supplied a
persistent output buffer without restoring that initialization and also added a
cross-call validity lifetime. Stale physical indices or an invalid reuse
lifetime are therefore the leading mechanisms. The end-to-end A/B proves the
candidate is wrong, but does not distinguish those two mechanisms yet.

A future retry must first prove fully initialized row contents and per-layer,
per-scheduler-step lifetime on a small CUDA oracle. Allocator removal alone is
not an acceptable reason to weaken those contracts.

## Fresh control matrix

The hardened runner completed with exit code zero:

| Case | TTFT (s) | Effective prefill (tok/s) | Per-request decode (tok/s) | Wall (s) |
| --- | --- | --- | --- | ---: |
| Medium C1 | 15.62 | 1011.9 | 27.47 | 17.92 |
| Medium C2 | 17.96 / 29.01 | 880.1 / 544.9 | 27.55 / 28.91 | 36.19 |
| Large C1 | 31.73 | 1014.1 | 27.46 | 34.03 |
| Large C2 | 35.53 / 61.18 | 905.7 / 525.9 | 24.90 / 30.24 | 68.27 |

All six control requests completed 64/64 tokens, retained their own marker,
contained no peer marker, and reported no request error.

## Other current upstream candidates

Live GitHub state was rechecked on 2026-09-02 before making these decisions.

| Candidate | Decision for this appliance |
| --- | --- |
| vLLM #53109 packed RMSNorm outputs | Already ported and accepted as `d3860af`; fresh isolation matrix above passes. |
| vLLM #54394 TP prefill row sharding | Ported and rejected at `36a42f1`; cross-node row exchange loses on this target. |
| vLLM #49678 physical-index reuse | Ported and rejected at `c23cbc0`; output corruption. |
| vLLM #54374 DFlash sliding-window AOT | No-op here: the GB10 drafter selects FlashAttention 2, and the affected AOT schedule is already disabled unless FlashAttention 3 is active. |
| GLM short-decode fast path `f221389f` | Outside the frozen 16K/32K prompt target; it applies only at sequence lengths at or below the 2,048 index top-k. |
| vLLM #53878 fused sparse-Q concatenation | Wrong active backend; it changes `FLASHMLA_SPARSE`, while this appliance runs `FLASHINFER_MLA_SPARSE_SM120`. |
| vLLM #53785 dense/masked MHA | Its optimized enablement is for SM100, not GB10 SM121. |
| vLLM #52657 FlashInfer MLA prefill sync | Changes dense MLA prefill, not the active sparse SM120 path. |
| vLLM #53425 sparse-MLA block size | DSv4-specific and not a GLM-5.3 port candidate. |
| vLLM #52555 custom all-reduce size | Same-node TP2 only; the appliance is cross-node TP2 over CX-7. |
