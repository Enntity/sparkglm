# Rejected EXL3 prefill fusion experiments

**Date:** 2026-09-03
**Status:** measured and rejected; neither candidate is enabled by default

## Question

Can the grouped GLM-5.3 EXL3 fat-expert path become materially faster by
avoiding the FP32 gate/up activation arena between the direct-trellis GEMMs and
SwiGLU?

The shipped path writes gate and up as FP32, launches a phase-wide activation
kernel, then writes the FP16 down input. That makes fusion attractive: at one
routed row it accounts for 16 KiB of gate/up write+read traffic before the 2 KiB
down input is produced.

## Variant A: fuse up projection with activation

This version materialized only the FP32 gate half. The up-projection tile stayed
in shared memory and immediately performed the exact clamp, rounded SwiGLU, and
down-input Hadamard.

Model-free results were bit-for-bit exact. The candidate beat the shipped
grouped kernel by 2-6% on several large, uniform shapes and 6.3% on an initially
chosen skewed route list. The complete model contradicted that proxy: an
exact-salt ~33K-token C1 took 24.008 s versus the accepted 23.479 s baseline,
2.3% slower. The activation work lengthened the up-projection kernel's critical
path and its synthetic route mix was not representative enough.

## Variant B: fully paired gate+up+activation tile

This version loaded an M64 activation tile once, decoded gate and up together,
kept both FP32 accumulators on chip, applied exact SwiGLU/Hadamard, and emitted
only the down input. It removed the global gate/up arena and reduced grouped
execution to gather, paired projection/activation, and down/scatter.

The kernel remained bit-for-bit exact and beat the shipped grouped path by
3.3-5.5% across the model-free shapes. Resource inspection supplied the catch:
the paired kernel used 126 registers/thread versus 64 for the shipped projection
kernel. That halves resident blocks on GB10 and consumes most of the theoretical
traffic win.

After long-prefill JITs were explicitly warmed and excluded, the complete-model
results were:

| ~33K actual tokens | Shipped grouped | Paired fusion | Change |
|---|---:|---:|---:|
| C1 TTFT median | 23.479 s | 23.589 s | 0.47% slower |
| staggered C2 TTFT p50 median | 38.850 s | 38.340 s | 1.31% faster |
| staggered C2 TTFT p95 median | 47.053 s | 46.748 s | 0.65% faster |
| staggered C2 wall median | 47.554 s | 47.249 s | 0.64% faster |
| staggered C2 aggregate prefill | 1,387.7 tok/s | 1,396.6 tok/s | 0.65% faster |

The C2 gain is too small to justify a C1 regression, extra code, and another
production branch. This variant was rejected.

One earlier 25.335 s C1 observation is deliberately excluded: the live JIT
monitor recorded `mhc_pre_big_fuse_with_norm_tilelang` compiling during that
request. The discarded warmup had also compiled `_kpool_tail_seed_kernel` and
`_prepare_dflash_inputs_kernel`. A server reporting healthy is not sufficient
evidence that long-prefill shapes are warm.

## Variant C: split exact SiLU and up multiplication

To retain the 64-register projection shape, this version computed clamped SiLU
in the gate epilogue, stored that exact FP32 result, and left only clamped up
multiplication plus Hadamard in the up epilogue. Splitting at an FP32 operation
boundary preserves the shipped arithmetic.

It was bit-for-bit exact and won 5-9.5% for very large uniform experts, but lost
1.4% on the production-like mixed fat-expert list and 1.7% at 16 experts x 256
rows. It failed the model-free promotion gate and was never loaded with the full
model.

## Route-reuse finding and next large lever

A diagnostic-only route trace over long C1 and staggered C2 prefills found:

- consecutive chunks repeated the layer's top expert 68.8% at C1 and 66.2% at
  C2;
- caching the previous top expert would cover 7.84% and 7.41% of routed rows;
- an oracle current-top-expert cache would cover only 8.31% and 8.05%.

Thus prediction is not the hard part: the previous chunk captures almost all of
the available top-one locality. A model-free dense-cache probe showed that a
persistently reconstructed FP16 expert projection beats direct EXL3 by about
22-40% through the relevant 256-1024-row range while remaining numerically
equivalent at the measured tolerance. Reconstructing every call loses, and an
FP8 cache is faster but is not quality-equivalent enough to promote without a
model-quality gate.

The next large experiment is therefore a device-resident, previous-hot-expert
cache with asynchronous replacement and a dynamic-M tensor-core kernel. It must
skip the cached expert in the grouped EXL3 planner without copying counts to the
host. Its likely ceiling is several percent of complete prefill—not the 20-40%
projection-local number—because one cached expert covers roughly 8% of routes.

## Provenance

The three orchestration/fusion variants and the route-cache experiment design
are original work in this repository. Their direct K4 trellis arithmetic,
Hadamard helpers, and tensor-core fragment layout remain derived from the exact
ExLlamaV3 and Reederey revisions recorded in `docs/PROVENANCE.md`. No external
gate/up fusion or hot-expert cache implementation was copied.
