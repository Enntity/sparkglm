# EXL3 32-row fat-expert tile: rejected

Date: 2026-09-03

## Verdict

Reject M32. It is bit-exact and reduces the compiled kernel from 64 to 47-48
registers per thread, but it is only neutral at a few small boundaries and is
2-20% slower across the rest of the measured production range. The candidate
was never wired into serving.

The accepted M64 kernel remains the best measured row tile. No further M-tile
search is justified without a materially different dataflow.

## Candidate and method

The candidate retained K16, N128, the 256-thread CTA, the three-stage
`cp.async` pipeline, K4 MCG decode, FP32 accumulation, Hadamard epilogues and
scatter semantics. It changed only the M tile from 64 to 32, halving live
accumulator state and tail granularity while increasing CTA count and repeated
A-tile loads.

- Source base: `27cb29aa75`
- Candidate image: `sparkglm-vllm:exl3-m32-candidate-20260903a`
- Head image ID: `43317a8f1a66a2c28ce79fe46270151eada4c3f43e88ff8ff74723cdc0910d64`
- Worker image ID: `a08501d2c495b2e0098896f626a8e0631e7baab1417cb6164d0ea1940a85dad8`
- Shape: hidden 4,096, TP-local intermediate 1,024
- Timing: CUDA events, 20 warmups, 100 iterations, five alternating-order
  repeats; median
- Timed pipeline: paired gate/up, fused SwiGLU/Hadamard and down/scatter

Every tested M32 gate/up tensor, activation tensor and final scattered output
was bit-identical to M64 (`max_abs=0`). Delta is M32 versus accepted M64;
positive is slower.

| Expert rows | M64 (ms) | M32 (ms) | Delta |
| ---: | ---: | ---: | ---: |
| 129 | 0.1167 | 0.1191 | +2.1% |
| 145 | 0.1147 | 0.1167 | +1.8% |
| 192 | 0.1200 | 0.1287 | +7.2% |
| 255 | 0.1744 | 0.1745 | +0.05% |
| 257 | 0.1743 | 0.1744 | +0.07% |
| 320 | 0.1878 | 0.2062 | +9.8% |
| 383 | 0.2009 | 0.2316 | +15.3% |
| 385 | 0.2355 | 0.2561 | +8.7% |
| 512 | 0.2846 | 0.3341 | +17.4% |
| 640 | 0.3499 | 0.3997 | +14.2% |
| 768 | 0.3885 | 0.4657 | +19.9% |
| 1,024 | 0.5422 | 0.6311 | +16.4% |
| 2,048 | 1.1601 | 1.2955 | +11.7% |
| 4,096 | 2.3226 | 2.6186 | +12.7% |
| 6,528 | 3.7856 | 4.2035 | +11.0% |

## Consequence

M64 is the balance point for this GB10 pipeline: M128 carries excess
accumulator and tail cost, while M32 repeats too much A movement and CTA
epilogue work. The next expert-compute work should use M64 and change the work
inside the tile—packed-codebook decode, Hadamard/output traffic, or overlap—
rather than changing the row tile again.
