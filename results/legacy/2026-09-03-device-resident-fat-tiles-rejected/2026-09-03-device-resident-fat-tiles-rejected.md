# Persistent stock-kernel fat tiles: rejected operator prototype

Date: 2026-09-03

## Verdict

Reject before full model load. The prototype removed the routing-count D2H
synchronization and Python per-expert dispatch, but it reused ExLlamaV3's
small-row persistent MoE microkernel for oversized experts. At production
hidden/intermediate dimensions it was 35-49% slower than the accepted E2 path.

The runtime feature and kernel patch were removed. The complete-layer
microbenchmark remains as the gate for the next device-resident dispatcher.

## Candidate

The stock `exl3_moe` kernel normally skips experts with more than 128 routed
rows. An additive fat-only mode made each persistent expert group consume an
oversized expert in bounded 128-row slices. Python launched the ordinary thin
mode once and the fat-only mode once, without reading expert counts on the CPU.

This preserved bounded scratch and passed the full synthetic GPU overlay suite,
including all-fat routing, mixed thin/fat composition, expert-map behavior,
CUDA graph capture, and finite output checks. Relative maximum output deviation
from the accepted E2 path was approximately 0.1% on the production-shape
microbenchmark, consistent with different accumulation ordering.

## Complete-layer result

Command shape: 64 K4 MCG experts, top-8 routing, hidden width 4,096, TP-local
intermediate width 1,024, Zipf-skewed distinct experts per token. Timings include
mapping, sorting, count construction, thin experts, fat experts, activation,
weighted scatter, host synchronization, and Python launch gaps.

| Tokens | Fat experts | Maximum expert rows | Accepted E2 (ms) | Persistent tiles (ms) | Candidate / accepted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 7 | 450 | 6.459 | 9.903 | 0.652x |
| 2,048 | 35 | 1,803 | 16.964 | 33.120 | 0.512x |

Both rows were measured after one warmup with two timed iterations on one idle
GB10. The loss is large enough that a 288-expert or endpoint run cannot reverse
the decision and would only spend a full model reload.

## Cause and next design constraint

The persistent ExLlamaV3 kernel assigns eight SMs to an expert and advances its
GEMMs in 16-row chunks. That is effective for thin/decode experts but leaves
large-row work less parallel than the accepted E2 kernel, whose 128x128 tiles
can occupy the whole GB10. Eliminating host dispatch does not compensate for
moving fat-expert arithmetic back onto the small-M kernel.

The next implementation must therefore retain the accepted paired/fused E2
microkernels and replace only their scheduler: build compact `(expert, row
tile, output tile)` jobs on the GPU and let persistent CTAs consume those jobs.
It must first beat `benchmark_exl3_fat_dispatch.py` at 512, 2,048, and 7,168
rows before earning a model load.
