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

## Follow-up after first screen

The universal dispatch screen failed on small shapes; see
[the recorded screen](../../../results/candidates/2026-09-07-e3-screen/README.md).
`Dockerfile.runtime` adds an original source-hash-locked adapter, disabled unless
`SPARKGLM_EXL3_E3=1`. It preserves reference dispatch below 4096 input tokens and
excludes the nonlocal sentinel from E3 pointer tables. This is a new hypothesis,
not a passed performance gate. Kernel code and extracted tables remain MIT;
the adapter and insertion tooling are original Apache-2.0.

The top-8 cardinality follow-up rejected the 2048-token threshold (3–7%
planner regression). The next policy raises the threshold to 4096; balanced
4096 rows were neutral and 7168 rows improved in the first screen. This
threshold change requires its own integration evidence.

The three-pair full-model sweep rejected broad >=4096 dispatch: C4 improved,
but 16K solo work regressed and the primary 32K C2 effect was noisy. The next
candidate adds `SPARKGLM_EXL3_E3_POLICY=concurrent`. It reads CPU attention
metadata and uses E3 only when a prefill shares the batch with another request.
Solo prefill keeps the reference kernel. Dummy profiling reserves E3 buffers
and then executes the reference path to account for both sets of persistent
scratch before cache sizing; unknown metadata and DBO
lists fall back to the reference. `SPARKGLM_EXL3_E3_TRACE=1` records the first
profile-reserve/selected/reference decisions for integration qualification. This new
policy is unqualified until its own tinyGLM and full-model runs complete.

The follow-up 32-row expert threshold is a distinct opt-in policy. Its top-8
operator sweep showed gains already at 2048 input tokens, so that threshold
uses E3 from 2048; other expert thresholds retain the 4096 guard. Concurrent
selection still protects solo prefill. The new combination needs its own
TP2 fixture and full-model qualification; the rejected broad policy remains
rejected and the default remains disabled.

The original concurrent adapter profiled only E3. A later solo prefill could
therefore allocate reference scratch that the automatic KV calculation had
not seen. The corrected adapter reserves E3 scratch and returns to the
reference profiling path. This changes allocation accounting, not the copied
E3 numerical kernels. The focused CPU regression checks reservation, reuse,
reference fallthrough, and forbidden growth during graph capture. It requires
fresh TP2 integration evidence before recommendation; source checks alone do
not establish a performance gain.

The [completed corrected-profile campaign](../../../results/candidates/2026-09-07-e3-profilefix-full/RESULT.md)
retains three full matrices for concurrent dispatch, 32 expert temporary rows,
7168-token chunks, and the BF16 DFlash2 draft. This is the EXL3 finalist from
this campaign; bounded arithmetic failures still prevent a general-quality
or release claim. The corrected broad-policy retest failed its tiny throughput
guard and did not receive another full-model load.

Build `Dockerfile` over the verified non-MXFP8 reference, then apply
`Dockerfile.runtime` to that E3 image. Pin both resulting image IDs and source
revision. LLooM materialization uses `--e3 --e3-policy concurrent
--exl3-temp-rows 32 --prefill-tokens 7168` with the ordinary BF16 draft.
Stop the active appliance runtime before replacing its managed profile.
