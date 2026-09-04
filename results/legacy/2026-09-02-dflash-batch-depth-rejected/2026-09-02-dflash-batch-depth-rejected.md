# DFlash2 batch-depth rejection — 2026-09-02

## Verdict

Reject the batch-size schedule `C1=K7, C2=K4, C3-C4=K2`. It reduced both
verification width and actual DFlash2 proposal work, but lost on the target
medium/large staggered C2 workload. The appliance was restored to fixed K7.

The experiment's runtime changes are not retained. The benchmark now records
the SHA-256 of every input prompt so later A/B claims can prove byte-identical
workloads rather than relying only on a human-readable salt.

## What was tested

The stock dynamic-speculative scheduler truncates the number of draft tokens
verified by the target, but the V2 DFlash2 proposer still constructs and runs
the full maximum-K query. A candidate therefore carried the active K through
the DFlash/DFlash2 proposal path, input preparation, selector sampling, output
handoff, and CUDA-graph dispatch.

GPU startup proved that the candidate captured ten target and ten DFlash2
FULL graphs spanning the K7/K4/K2 query shapes. Fixed K7 captured four DFlash2
FULL graphs. Candidate graph memory was 1.28 GiB versus 1.59 GiB for fixed K7.
All 20 bounded C1-C4 boot warmups passed.

Both A/B runs used the same two-Spark TP2 service, phase interleave q8/p2048,
DFlash2 draft TP1, EXL3 target weights, FP8 target KV, five-second C2 stagger,
and 400 forced output tokens per request. Every paired input SHA-256 matched;
all requests completed, retained their own isolation marker, emitted no peer
marker, and returned no error.

## Byte-identical A/B

| Case | Fixed K7 aggregate decode (tok/s) | Dynamic aggregate decode (tok/s) | Fixed/dynamic B TTFT (s) | Fixed/dynamic wall (s) | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Medium C2, 15,810 + 15,807 prompt tokens | 22.188 | 20.963 | 34.479 / 36.311 | 53.825 / 57.753 | reject: decode -5.5%, B TTFT +5.3%, wall +7.3% |
| Large C2, 32,182 + 32,179 prompt tokens | 16.737 | 16.380 | 68.022 / 70.238 | 83.089 / 85.197 | reject: decode -2.1%, B TTFT +3.3%, wall +2.5% |

C1 remained K7 in both modes and served as a regression/noise guard. Medium
C1 measured 43.870 versus 40.121 tok/s; large C1 measured 29.247 versus 32.704
tok/s. This spread is why acceptance is based on both C2 cases and all three
user-visible metrics, not a favorable single-stream sample.

## Why reducing work still lost

K7's late positions are useful often enough to amortize their cost. In the
fresh fixed-K7 control, 577 draft rounds proposed 4,039 tokens and accepted
1,902; accepted counts by position were 469, 364, 300, 245, 205, 173, and
146. Cutting to K4 discards the last three accepted-token opportunities and
requires more target rounds to produce the same 400 tokens per request.

On GB10, shrinking a low-row decode forward does not reduce step time in
proportion to token rows: fixed launch, collective, expert-routing, and kernel
costs remain. The additional rounds outweighed the smaller DFlash and verifier
matrices. This also explains why scheduler-only truncation was worse: it paid
the full K7 proposal cost while giving up K7's accepted-token yield.

The result does not rule out request-specific adaptive depth. A future attempt
must use DFlash2 selector confidence or measured marginal acceptance to avoid
late positions only on low-value requests, then bucket requests into graphable
shapes. A static batch-size cutoff throws away high-confidence continuations
indiscriminately and is not a worthwhile appliance optimization.
