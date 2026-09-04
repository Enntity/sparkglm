# EXL3 K4 warp-shuffle decode: rejected

Date: 2026-09-03

## Verdict

Reject. Replacing each lane's second shared-memory packed-word load with a
shuffle from the preceding lane is bit-exact, but it is not a regression-free
or material improvement. The full M64 expert pipeline regressed at three of 15
production row counts and was effectively neutral at the largest shapes.

The candidate was never wired into serving or loaded with model weights.

## Candidate

The inherited K4 decoder has every lane load its packed word and the preceding
lane's packed word before extracting eight overlapping 16-bit codebook inputs.
Across a warp, that loads the same 32 shared words twice. The candidate loaded
one word per lane and used `__shfl_sync` to obtain the preceding word. It kept
the same funnel shift, bit extractions, `decode_3inst_2` calls, BF16 fragments,
MMA order, accumulation, Hadamard, scaling and scatter arithmetic.

Both paired gate/up and down/scatter used the candidate decoder. Final gate/up,
activation/Hadamard and scattered outputs were bit-identical to accepted M64 at
every tested shape (`max_abs=0`).

## Production-shape sweep

The high-confidence sweep used one GB10, hidden width 4,096, TP-local
intermediate width 1,024, K4 MCG trellises, 40 warmups, 250 timed iterations,
15 alternating-order repeats and medians. Speedup is accepted M64 divided by
the candidate.

| Rows | Accepted M64 (ms) | Warp decode (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 129 | 0.11536 | 0.11310 | 1.0200x |
| 145 | 0.11696 | 0.11488 | 1.0181x |
| 192 | 0.12270 | 0.12143 | 1.0104x |
| 255 | 0.17361 | 0.16803 | 1.0332x |
| 257 | 0.17532 | 0.17177 | 1.0207x |
| 320 | 0.19157 | 0.19190 | **0.9983x** |
| 383 | 0.21049 | 0.20555 | 1.0240x |
| 385 | 0.24193 | 0.24032 | 1.0067x |
| 512 | 0.28937 | 0.29028 | **0.9968x** |
| 640 | 0.36650 | 0.36252 | 1.0110x |
| 768 | 0.40909 | 0.40447 | 1.0114x |
| 1,024 | 0.56872 | 0.56053 | 1.0146x |
| 2,048 | 1.17054 | 1.18189 | **0.9904x** |
| 4,096 | 2.42765 | 2.42561 | 1.0008x |
| 6,528 | 3.87458 | 3.86115 | 1.0035x |

## Component isolation

A seven-repeat component probe did not reveal one consistently winning half.
At 2,048 rows it measured pair gate/up 0.5504/0.5449 ms and scatter
0.6059/0.5898 ms (accepted/candidate), while the high-confidence composed
sample above regressed. At 512 rows both isolated candidate components were
slower. At 6,528 rows both improved by less than one percent and the complete
pipeline was tied.

## Consequence

The duplicated shared packed-word load is not the material bottleneck. Do not
fit a row-count selector to these small, unstable deltas. A useful next EXL3
kernel must change the amount or overlap of K4 dequantization, tensor-core work,
cross-warp Hadamard exchange or expert-weight reuse rather than substituting one
shared load with one shuffle.
