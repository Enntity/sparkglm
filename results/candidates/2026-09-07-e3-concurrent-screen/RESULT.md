# Concurrent E3 and the 32-row threshold

This remains an experiment. The kernel checks pass, but the TP2 tiny decode guard fails. No default or full-model performance claim is made.

| Nine-run tiny case | Reference tok/s | Candidate tok/s | Candidate/reference |
| --- | ---: | ---: | ---: |
| decode_c1 | 220.23 | 203.97 | 0.926 |
| mixed_c4 | 340.66 | 361.60 | 1.061 |
| long_c2 | 80.27 | 78.29 | 0.975 |

The original cap128 policy and the cap32/16K-chunk candidate both preserved exact token IDs. The three-run and nine-run observations are all retained; the longer set was declared before execution to investigate observed timing variation. The regression persists and needs an identical-image E3-off control.

Unsanitized cap32 top-8 kernel speed ratios (reference/candidate):
- M2048 balanced: 1.209x
- M2048 skewed: 1.442x
- M4096 balanced: 1.479x
- M4096 skewed: 1.430x
- M7168 balanced: 1.523x
- M7168 skewed: 1.355x

Numerical checks use rtol=1e-4 and atol=1e-5. Graph replay changes activations and route weights, and both implementations retain output redzones. Sanitizer timings are instrumentation artifacts and are not performance samples.

Policy traces show profile, reference, and concurrent selection on both ranks. G1 calls the shared extension directly; the later image changes the guarded minimum to 2048 only for cap32. Both new rank images have matching source hashes and inherit the same extension.

The first sanitizer invocation failed because the executable was absent from the image PATH; the successful retry mounted the installed CUDA 13.0 sanitizer. No kernel failure is inferred from that invocation error.
