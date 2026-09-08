# Humming fixture screen

The installed backend selected HUMMING and completed the three warmed TP2 repetitions with stable token IDs. Initial mixed-C4 warmup was nondeterministic, as in other cold fixture runs. This is an unqualified backend exploration, not a model performance or quality result.

| Case | Median output tok/s | Median wall seconds |
| --- | ---: | ---: |
| decode_c1 | 138.19 | 1.852 |
| mixed_c4 | 265.79 | 1.926 |
| long_c2 | 78.12 | 0.819 |

Timing varied substantially across the short repetitions; all individual runs are retained. These results do not favor Humming over the native CUTLASS fixture, but tiny geometry cannot rank full-model performance.

The installed humming/schema/base.py get_fallback_input_dtype deliberately maps float4e2m1 activations to float8e4m3. This path reads NVFP4 weights but is not evidence of native W4A4 execution. The native CUTLASS probe remains the separate FP4 hardware proof.
