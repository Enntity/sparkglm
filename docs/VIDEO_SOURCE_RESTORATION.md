# Restoring the posted-video implementation

## Why the previous reconstruction was wrong

The capture says `4b23759+c805318`. Those are recipe/overlay commits, not a
complete description of their base image. Their Dockerfile inherited a custom
vLLM foundation at `6a49e49e7e6a3226197a2ceefcf217cdf55f751e`, then layered the
M64 EXL3 work and grouped-prefill/cooperative-decode implementation on top.

The first release reconstruction instead started with the pinned Mia-compatible
vLLM image at `487ecf187d3dfe74d2cf6119a92881dba403c219`. Matching the final
overlay and flags left inherited runtime changes behind. The earlier
`2026-09-04-clean-release-c4-replay` bundle measures that incomplete build;
it does not establish parity with the video implementation.

## Exact restoration boundary

- `patches/video/native.patch`: the complete native-source difference from
  vLLM `487ecf187d3dfe74d2cf6119a92881dba403c219` to the historical foundation.
  It restores FP16 top-k dispatch and DeepGEMM paged-MQA FP16 support.
- `patches/video/runtime.patch`: seven substantive Python-file differences,
  captured from the retained image and applied after the current recipe
  patchers. This restores the indexer, scheduler, cache coordinator, and FA
  discovery code rather than reimplementing the features from descriptions.
- EXL3 fat-GEMM, cooperative-decode, their headers, and the Python dispatch
  match `c805318` except for corrected license headers. The actual historical
  compile inputs are under `/opt/glm53/pair/`, not the stale predecessor under
  `/opt/glm53/exl3-fat-kernel/`.
- Subsequent launcher safety, public chat-template integration, attribution,
  and test/documentation work remain. These are explicit non-engine changes;
  this is not a claim that every repository or container byte matches.

The reference image is
`sha256:0b17bd9246763d74e2f5e1b79fecdcb6a8ef03e1b8e5823f2d2183ceafb91159`.
The manifest records all 2,423 installed vLLM Python hashes, the eight changed
native-source hashes, and only three allowed nonfunctional file differences.
Unknown changed, missing, or additional Python files fail the Docker build.
The separate `--kind video-runtime` check verifies a running container after
startup, requiring the recorded 16 ms spinwait transformation as well. It is
intentionally specific to the video profile, not arbitrary operator overrides.

The retained build-version exception is not an assertion that version strings
are unused: vLLM includes the version in compilation/configuration cache
fingerprints and reported identity. Those fingerprints can differ on the
rebuild. The historical source's CUDA platform selection distinguishes CPU
builds rather than these two development revisions; the inspected EXL3 and
FlashInfer source does not dispatch on this vLLM version. Warm each image
independently; do not use cache reuse or identical startup time as a parity
claim.

The native build compiles `_C_stable_libtorch` and `_deep_gemm_C`. Other native
components are retained from the pinned base: their source is unchanged
between the base and historical foundation, although independently built
binaries can have different hashes. This is a source-equivalence boundary,
not a bit-identical binary claim. Full integration testing remains necessary.

## Provenance

This is a restoration, not a new optimization. The native sparse-selector work
is derived from vLLM's FP16 work, including Woosuk Kwon's
`839529e52230c649bd8e8d5117ff8ed773f68106`. The foundation carries the
Reederey recipe baseline at `b229968a64ae3a270acdda9ce539a421e21598d7`,
including scheduler alignment and per-group cache-retention support, plus
SparkGLM's phase-interleave integration. DeepGEMM is pinned to
`8b1392b978f5a03c828dd1711090d7fb50958b8a`. See `docs/ATTRIBUTION.md`
and the source ledger; do not attribute all inherited changes to SparkGLM.

## Validation status

The restored image passes its source gates and all embedded container tests.
Thirty native top-k cases (FP16/FP32, decode/prefill, graph replay) match the
reference's selected-value hashes. Native output order and choice among equal
boundary scores are not guaranteed; the oracle compares exact selected values
and verifies valid unique indices.

tinyGLM outputs match the retained video image for C1, staggered C4, and long
C2. The initial three-repetition C1 sample missed the 5% throughput guard
(5.03% lower); a retained ten-repetition follow-up passed all three cases.
Do not omit the initial failure or treat this as a performance improvement.
The tiny fixture launcher also needed its generated, shape-specific snapshot
revision instead of the obsolete hard-coded `tinyglm-v1` name.

The newer public chat template was checked separately: it renders identical
prompt bytes for the four recorded video requests. Its tool/None fixes remain
explicit later changes, not silently claimed to exist in the old template.

The [controlled ten-repetition follow-up](../results/candidates/2026-09-04-video-runtime-isolation/TEN_REP_RESULT.md)
found effectively tied headline speed under matched cache/CPU controls.
The earlier small-sample deficit did not reproduce. This is release
verification for the video workload, not a new speedup or quality claim.

The [restoration result](../results/candidates/2026-09-04-video-source-restoration/RESULT.md)
and [runtime-isolation audit](../results/candidates/2026-09-04-video-runtime-isolation/RESULT.md)
retain all partial results, source checks, environment differences, image
identities, and limitations. Source parity, package-environment parity, and
performance parity are distinct tests. None is a complete G3/G4/G5 certificate.

The replay client accepts `--cache-salt` to isolate repeated runs without
editing prompt text. The production profile does not expose the development
`/reset_prefix_cache` route. A fresh cache namespace supplies the same cache-miss
condition without enabling that route or reloading the checkpoint.
