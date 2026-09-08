# E3 comparison against SparkGLM grouped prefill

Unqualified, disabled experiment. Copies Mia E3 at the MIT-licensed revision
`2c0ebe55a91ac8c0868cd6cba264e14bce93c66b`, including its full-build SiLU
precision correction. The current SparkGLM reference remains unchanged.

Hypothesis: the four-stage K32 pipeline, decoded-weight reuse, fused gate/up
activation, and vector atomic scatter outperform our existing grouped M64
implementation. Mia's reported gain against its E2 host loop is not our gain.
Prior local paired fusion and direct-epilogue experiments were rejected;
this tests a different complete arithmetic schedule against that same reference.

Build the additive translation unit using the cached reference image. No vLLM
rebuild or full checkpoint load is needed. G1 compares both implementations
inside one image, alternating timed order, including device-table construction.
Use production TP-local H4096/I1024, boundary and skewed routing, graph replay,
and changed graph inputs. Retain all samples, source hashes and image ID.

Screen criterion: at least 3% lower geometric-mean latency, no reproducible
shape regression above 2%, finite outputs and reference-relative numerical
error within rtol=1e-4 / atol=1e-5. This tolerance accounts for changed FP32
accumulation order; it does not qualify model quality or activation rounding.
Then run tinyGLM, sanitizer checks, and exact-token G3/G4 through LLooM before
considering adoption. Primary endpoint metric: 32K staggered C2 completion wall
time, protecting C1, TTFT, stream gaps, capacity, semantics and cancellation.

Kernel files and builder are copied verbatim; `tables.py` extracts the upstream
table functions. `upstream.json` records copied hashes. MIT notices are retained
in LICENSE and the repository provenance ledger. Benchmark code is original
Apache-2.0 and calls the existing SparkGLM matrix fixture.
