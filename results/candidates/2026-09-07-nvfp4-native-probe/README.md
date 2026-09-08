# Native NVFP4 execution on GB10

A standalone synthetic H4096/I1024, 8-expert, top-8, 64-token W4A4 call executed
in the preserved SparkGLM image. The profiler records CUTLASS
`MainloopSm120ArrayTmaWarpSpecializedBlockScaled`, FP4 E2M1 operands and
SM120 block-scaled MMA. This is positive native FP4 execution evidence on GB10.

The finite-output numerical screen against an unquantized BF16 clamped
reference produced relative L2 0.2654 and cosine 0.9652, inside the deliberately
loose exploratory thresholds 0.30 and 0.96. These are NOT model quality bounds,
and this probe does not prove bitwise correctness, backend selection for every
model layer, CUDA graph correctness, or a performance win. A quantized exact
reference and full-model quality tests are still required. Timing is omitted:
compilation and profiler instrumentation are not benchmark measurements.
