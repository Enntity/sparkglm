# NVFP4 Marlin candidate

This is W4A16: NVFP4 weight storage with BF16 activations. It is not native FP4 arithmetic. The installed generic Marlin warning about missing hardware FP4 support describes its fallback path; it does not establish GB10 hardware capabilities.

The original probe quantizes synthetic weights to unswizzled FP4, independently decodes their E2M1 values and group scales, and compares the clamped dense reference to the installed vLLM repacking/Marlin path. Production dimensions H4096/I1024, E288, top8 and clamp10 were tested at M1/17/64/65/257/2048. All relative L2 errors meet 0.03; redzones and changed-input graph replay pass. M65/E16 compute-sanitizer reports zero errors.

The first probe attempted to mutate a frozen activation configuration after the initial numerical assertion passed. That harness error is retained; the corrected probe creates a fresh configuration and completes the clamp and graph checks.

| Tiny case | Median aggregate tokens/s |
| --- | ---: |
| decode_c1 | 231.01 |
| mixed_c4 | 385.09 |
| long_c2 | 79.68 |

All three warmed repetitions are deterministic. Cross-backend token hashes differ, as expected for the deliberate W4A16 versus W4A4 computation change. This is a candidate for full-model time, not a model-quality or performance-winner claim.
