# EXL3 pre-dequantized hot-expert cache: rejected

Date: 2026-09-03

## Verdict

Reject before model integration. Pre-dequantizing selected layer-experts once
into persistent FP16 weights produces exact outputs, but the complete cached
expert body is 26-85% slower than the accepted M64 K4 trellis path across the
production row range. Do not spend appliance memory or a model load on this
design.

## Hypothesis and gate

One TP-local routed expert occupies 24 MiB when its gate, up and down matrices
are reconstructed to FP16. A bounded cache of repeatedly selected experts could
fit in the remaining unified memory and use the native dense GEMM path while
the rest of the model stays EXL3.

`benchmark_exl3_hot_expert.py` tests the crossover without loading model
weights. Reconstruction occurs once and is excluded from timing. Both paths
include gate/up, output Hadamard/scales, exact rounded SwiGLU plus down-input
Hadamard, down projection, output Hadamard/scales, route weighting and scatter.
The accepted control uses the M64 paired gate/up and M64 down/scatter K4
kernels. The cached path uses ExLlamaV3's FP16 `hgemm` and existing transforms.

## Result

One idle GB10, hidden 4,096, TP-local intermediate 1,024, 10 warmups, 40 timed
iterations and five repeats:

| Expert rows | M64 K4 | Cached FP16 | M64 / cached |
| ---: | ---: | ---: | ---: |
| 129 | 0.1131 ms | 0.1654 ms | 0.684x |
| 192 | 0.1210 ms | 0.2037 ms | 0.594x |
| 320 | 0.1875 ms | 0.2741 ms | 0.684x |
| 512 | 0.2934 ms | 0.3687 ms | 0.796x |
| 768 | 0.4096 ms | 0.5861 ms | 0.699x |
| 1,024 | 0.5473 ms | 0.8470 ms | 0.646x |
| 2,048 | 1.1745 ms | 2.0521 ms | 0.572x |
| 4,096 | 2.3538 ms | 4.3654 ms | 0.539x |
| 6,528 | 3.7828 ms | 6.8793 ms | 0.550x |

Every final output was bit-identical (`max_abs=0`, relative RMSE 0). This is
not an arithmetic-quality tradeoff or reconstruction-cost artifact.

## Consequence

The accepted K4 pipeline's lower weight traffic is more valuable than routing
these shapes through generic dense FP16 GEMM. A hot cache also consumes four
times the packed-weight footprint and adds residency policy complexity, so it
has no compensating appliance benefit.

Continue with an on-the-fly K4 design that changes packed-codebook decode,
tensor-core issue/dataflow or overlap while retaining the compact weights.
