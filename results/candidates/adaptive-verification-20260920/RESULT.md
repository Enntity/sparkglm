# Adaptive verification: bounded TP2 comparison

| Whole-workload seconds | Fixed K5 | Adaptive K2/4/5 |
|---|---:|---:|
| C2 x32K median,3 repetitions |56.03|52.94|
| C4 x16K median,3 repetitions |60.58|61.23|
| C8 x16K,one pair |105.90|109.75|
| C2 x16K,1s stagger,one pair |45.52|34.50|

The C2 median was5.5% lower; C4 was1.1% higher and C8 was3.6% higher. Individual C2 changes were -1.9%,-5.5%,+0.8%; C4 changes were +9.0%,-5.0%,+1.1%. The predeclared median threshold passed, but variability limits any broad speed claim. One staggered case improved24.2%; it is not a repeated estimate.

Both arms used identical candidate images and graph shapes with off/on switching in one load, alternating order, discarded warmups and isolated paired cache salts. Exact paired prompts/token counts were checked. Every request generated400 tokens. Bounded full-model JSON schema, tool calls, cancellation recovery and isolation checks passed; no request failures, preemptions or OOM were recorded. These were direct backend timings, excluding gateway/admission overhead.

The preliminary tiny-model run was inconclusive: speculation was disabled, and one C4 repetition differed. It is not an adaptive G2 qualification. No complete G2/G3/G4 matrix or G5/endurance qualification is claimed. The source is published as an opt-in preview, with defaults unchanged.

[Source and profile](../../../research/adaptive-verification/README.md). [Model attribution](../../../docs/QUANT_ATTRIBUTION.md). This comparison uses NVIDIA NVFP4/MXFP8; the quant notice also covers historical EXL3 results elsewhere.
