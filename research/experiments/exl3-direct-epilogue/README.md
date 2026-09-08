# Warp-owned direct EXL3 epilogue

**Status: rejected by the model-free performance screen.** Both variants
compiled and passed the tested arithmetic/graph checks, but were slower.
The reference build and recommended defaults are unchanged. See the
[checksum-bound results](../../../results/candidates/2026-09-04-exl3-direct-epilogue/RESULT.md).

## Hypothesis

The grouped M64 tile currently exchanges fragments through shared memory,
computes each row's Hadamard/scaling into that same shared memory, then asks
other threads to read it back and write the final destination. This experiment
keeps the transformed row in its owning warp and writes directly instead.

Per active row/projection it removes a 512-byte shared write and 512-byte
shared read. Each 16-row epilogue needs two rather than three block barriers.
It does **not** remove the initial fragment exchange, weight reconstruction,
Hadamard arithmetic, global output bytes, or cross-expert atomic accumulation.
The last barrier remains mandatory for the next row tile or persistent task.

This differs from the rejected vector-store experiment: that experiment still
read transformed values back from shared memory. The direct path removes the
intermediate publication and its barrier; it is not a padding experiment.

The standalone M64 arithmetic oracle, K4 decode, MMA loop, grouped task planner,
activation, and cooperative decode kernels are untouched. The candidate changes
only the grouped tile's epilogue and adds a helper with the reference Hadamard
statement order. Risks include extra registers, occupancy loss, different
atomic contention, alignment errors, and tail-row synchronization hazards.

## Preparation and later build

These commands are CPU-only and do not contact the Sparks:

```bash
python3 tests/test_direct_epilogue.py
python3 research/experiments/exl3-direct-epilogue/build.py --print-dockerfile
```

On a **reserved** Spark with at least 32 GiB available, after committing the
reviewed experiment, build the separate candidate:

```bash
python3 research/experiments/exl3-direct-epilogue/build.py
```

The generated recipe reuses the normal source build's cached stages and changes
only the EXL3 compile stage onward. It does not avoid an EXL3 extension rebuild.
`sources.json` declares the exact changed source hash; every other reference
source check remains enabled. Never edit the frozen parity manifest to pass it.
No helper starts/stops servers, modifies `.env`, or loads checkpoint weights.

`Dockerfile.incremental` offers a second, local-only path when the qualified
reference image is already present: it verifies the inherited Python and EXL3
sources, applies the same declared CUDA change, and rebuilds only ExLlamaV3.
Record the base image ID before/after building; a Docker tag is not immutable.
This path inherits the reference's native binaries rather than rebuilding them.
It retains compiler objects for later iterations and is not an audited public
binary release. Its setup derives from the root Mia/ExLlamaV3 build recipe.

`Dockerfile.control` restores and recompiles the reference CUDA source using
that same compiler tree (about 30 seconds in this experiment). This separates
source changes from inherited-versus-rebuilt extension effects.
`Dockerfile.followup` builds the coalesced-atomic variant using the same cache.
It needs the initial candidate image and does not change its existing tag.
All three recipes are isolated experimental builds, not deployment targets.

## G1 gate when hardware is available

Run `bench.py` from an image with the repo mounted at `/workspace`, passing the
actual `docker image inspect` image ID and installed CUDA-source SHA-256. Its
`--help` documents flags. Run the identical script in the reference image and
the candidate image, retaining the JSON outputs outside Git until reviewed.

The supplied runner pins image IDs, validates installed source hashes, refuses
an existing output directory, and alternates three pairs with matching seeds:

```bash
SPARKGLM_GPU_RESERVED=1 bash research/experiments/exl3-direct-epilogue/run-g1.sh \
  sparkglm:local sparkglm-candidate:direct-epilogue /tmp/direct-epilogue-g1
```

This opt-in assertion is not an automatic reservation or contention detector.
Confirm the GPU is idle and keep resource/clock telemetry separately. The
runner cannot stop other services and does not start the full model.

For the coalesced follow-up, use `SPARKGLM_EPILOGUE_VARIANT=coalesced` and
`sparkglm-candidate:coalesced-epilogue`; the runner checks
`sources-coalesced.json`. `summarize.py OUTPUT_DIRECTORY` reproduces the paired
summary. `kernel_profile.py` is a separate diagnostic: profiler-perturbed
timings must never replace unprofiled A/B receipts.

The default suite covers cap=128 no-work behavior, 129..2048-row M64 boundaries,
zero/thin/fat mixed experts, overlapping routes, graph replay, output red zones,
exact gate/up and activation intermediates, and exact single-expert outputs.
Overlapping-expert output uses the existing `rtol=1e-5, atol=1e-6` contract
because floating-point atomic order can vary. Inputs are deterministic random
dummy trellises, not real checkpoint weights. Test multiple seeds and run CUDA
memcheck/racecheck separately; red zones are not a sanitizer substitute.

The serial M64 implementation is an **arithmetic oracle, not a speed baseline**.
Compare grouped graph timing to grouped graph timing between images. Alternate
reference/candidate process order across at least three pairs on idle hardware;
exclude compilation and warmup. Retain every sample and exact source/binary IDs.
These warm synthetic cases cannot model full-checkpoint weight-cache pressure.

Predeclared screening criterion: all correctness checks pass, at least 3% lower
geometric-mean grouped latency across active-fat cases, and no shape regresses
by more than 2% reproducibly. A borderline result is noisy, not a pass. Follow a
pass with the existing tinyGLM C1/C2/C4 gate and compute-sanitizer checks, then
the actual-token 16K/32K staggered full-model matrix. Primary endpoint metric:
32K C2 completion wall time, protecting C1, TTFT, visible event gaps, quality,
and memory capacity. No endpoint benefit is inferred from the G1 threshold.

## Provenance

Original SparkGLM work: warp-owned direct output schedule, source-locked
transformation, build declaration, and test harness. Hadamard/scaling operations
are copied unchanged from the published SparkGLM helper derived from
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) at
`c5d9c657966ffeeaa9353f0cc899f18629da4a13`. The unchanged M64 pipeline derives
from [Reederey](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) at
`0c03250cd7176a2fef9cbbf9329fed08c8750e7d`. Their notices remain in `LICENSES/`.
No upstream direct-epilogue implementation was copied. `patch.py` and generated
CUDA retain MIT AND Apache-2.0; original harness/build/test code is Apache-2.0.
