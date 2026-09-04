# FlashKDA prefill integration rejected

Date: 2026-09-03

## Verdict

Do not enable FlashKDA for GLM-5.3 by default. Revision
`3b225bf26bb8e218928a1fe14751cb48cf31d11b` is correct and 3.36-3.56x faster
than the Triton chunked KDA in the isolated two-sequence operator benchmark,
but the exact same-image endpoint A/B was neutral at large C2 and materially
worse at medium C2. The operator result does not compose into the current GLM
serving path.

## Corrected comparison

`benchmark_glm_flashkda.py` compares the current GLM Triton
`chunk_kda_with_fused_gate` path with FlashKDA. It uses the deployed TP-local
geometry: 32 heads, head dimension 128, BF16 inputs, FP32 state, gate lower
bound -5.0, and two variable-length sequences. It does not load model weights.

| Total tokens | Triton | FlashKDA | Operator speedup | Output relative RMSE | State relative RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 3.487 ms | 1.039 ms | 3.36x | 0.00561 | 0.00468 |
| 7,168 | 12.178 ms | 3.426 ms | 3.56x | 0.00561 | 0.00463 |

Both output and final state pass `torch.allclose` at 1% absolute and relative
tolerance. The two-part continuation check is stronger: splitting every
sequence on a 128-token boundary, exporting the first half's state, and resuming
the second half produces output and final state exactly equal to a one-shot
FlashKDA call.

## Independent oracle

The validated revision was built as a standalone exact-SM121 extension and
compared with FlashKDA's independent PyTorch recurrence oracle at fixed and
variable lengths, BF16 and FP32 state, and the exact GLM tensor shape. FlashKDA
matched that oracle bit-for-bit for output and final state. On the exact random
GLM case, the existing Triton result differed from the same oracle by relative
RMSE 0.00546 for output and 0.00402 for state.

## Why the first result was wrong

The initial harness reported an unwritten FP32 final state. That diagnosis was
false. The Triton `chunk_kda_with_fused_gate` implementation intentionally uses
its input `v` tensor as the output buffer. The harness invoked Triton first and
then reused the overwritten `v` as FlashKDA's input. Reversing call order or
giving each backend a private reset input eliminates the failure.

The corrected harness now:

- gives each backend a private `v` buffer;
- restores that buffer outside every timed CUDA interval;
- snapshots correctness outputs before timing mutates buffers;
- verifies that the final-state sentinel was fully overwritten;
- checks exact one-shot versus resumed continuation.

## Selected upstream revision

The prior vLLM pin was `b5d11010`. The selected dev revision `3b225bf` adds the
later performance and correctness work, including V-split execution, direct
FP32 final-state stores from resident fragments, an improved epilogue, and the
new Neumann inverse. The vLLM operator schema must also expose the two optional
checkpoint arguments added by that revision, although GLM serving does not use
them.

## Endpoint A/B

The candidate was overlaid onto the accepted M64 engine and compared with the
same image forced back to `--kda-prefill-backend triton`. Both modes used TP2,
DFlash2 K7 with draft TP2, FP8 KV, 7,168 scheduled tokens, four sequence slots,
work-conserving mixed scheduling, right-sized indexer workspace, 16 ms spin
wait, and a 10 GiB explicit KV reservation. Requests were five seconds apart
and forced 128 output tokens.

| Case | FlashKDA prefill | Triton prefill | Delta | FlashKDA decode | Triton decode | FlashKDA wall | Triton wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm medium C2 | 1,232.54 tok/s | 1,289.80 tok/s | -4.44% | 15.91 tok/s | 18.87 tok/s | 35.08 s | 30.44 s |
| Large C2 | 1,216.37 tok/s | 1,194.41 tok/s | +1.84% | 6.91 tok/s | 7.12 tok/s | 60.37 s | 60.00 s |

All eight measured requests completed 128/128 tokens, retained their own
marker, contained no foreign marker, and returned no API error. The first
FlashKDA pass was colder and slower still: medium C2 was 1,009.46 prefill tok/s
with 38.45 s wall, and large C2 was 1,276.27 prefill tok/s with 55.48 s wall.

The endpoint gate fails on medium wall time and decode delivery. The small
large-prefill throughput increase does not compensate for the regression and
does not reproduce the predicted 5-7% endpoint gain.

## Follow-up boundary

Before revisiting FlashKDA, benchmark the exact production N=1, BF16-state,
row-strided projection views and include adapter/workspace costs. Do not infer
endpoint benefit from the contiguous N=2 operator benchmark. The current
synchronized profile still orders direct EXL3 expert compute ahead of further
KDA work.

The first follow-up ruled out sequence count and state dtype: the vendored
vLLM operator at N=1 with BF16 state remained 3.47x faster at 2,048 tokens and
3.52x faster at 7,168 tokens, with exact split/resume continuation. A second
follow-up reproduced GLM's row-strided Q/K/V and beta projection views and
included the adapter's contiguous packing inside the timed interval. FlashKDA
still won 2.59x at 2,048 tokens and 2.67x at 7,168 tokens. Packing reduced the
kernel headline but does not explain the endpoint miss.

The remaining discrepancy is therefore whole-engine composition: interaction
with the full layer/TP schedule, synchronization not represented by isolated
CUDA-event timing, or an overestimated KDA share in the enclosing profile.
Re-profile the complete FlashKDA and Triton forwards before doing more adapter
work.
