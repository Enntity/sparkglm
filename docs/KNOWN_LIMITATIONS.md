# Known limitations

This is the short list a user, reviewer, or benchmark reader should understand
before treating SparkGLM as a finished appliance.

## Model-semantic approximation in sparse MLA

GLM-5.3's k-pool indexer can emit 2051 candidates: 512 pools expanded by four,
plus up to three recent tail tokens. The available FlashInfer SM121 sparse-MLA
kernel is instantiated at exactly 2048 candidates. The current compatibility
overlay expands 511 pools and preserves the recent tail, so as many as four of
the lowest-ranked pooled candidates are omitted.

That is a small candidate-set change (at most 4/2051 on affected rows), but it
is not mathematically exact model execution. The proper root fix is an SM121
kernel/dispatch contract that accepts all 2051 candidates, followed by output
and quality qualification. Until then, comparisons must use the same
approximation on both arms and disclose it.

## Qualification status

- No checked-in result currently gives this release candidate post-policy G5
  qualification.
- The retained performance campaign is `legacy`: it predates the exact 16K/32K
  prompt calibration, complete G3 matrix, and current G4 semantic requirements.
- The root Docker build is now statically checked for complete build-context
  inputs, but a clean ARM64/SM121 image build and two-rank boot still require
  the DGX Spark hardware gate.
- `scripts/full-model-gate.sh` captures exact tokenizer counts and fails on
  incomplete output or request-marker contamination. The operator must still
  alternate baseline/candidate order and create the checksum-bound
  qualification record; the script cannot restart two different engines by
  itself.

## Current-best optimizations and their limits

- `EXL3_GROUPED_PREFILL_K4=1` is part of the posted current-best configuration
  and defaults on. Its retained pre-policy evidence was positive, but it
  allocates roughly 1.2 GiB of persistent scratch per rank and has not passed
  the current full G3/G4 matrix. Set it to `0` for rollback.
- `EXL3_DECODE_COOP_K4=1` is also part of the posted configuration and defaults
  on. It passed exact-shape and tinyGLM checks, but its isolated long-C2 signal
  was flat and its broader semantic qualification is incomplete. Set it to `0`
  for rollback.
- The posted-video foundation used native FP16 sparse-selector logits, requiring
  a matching DeepGEMM build. The earlier release reconstruction omitted it.
  The restoration now builds those components together; source parity must
  not be confused with full-model numeric or performance qualification. See
  `docs/VIDEO_SOURCE_RESTORATION.md` for current verification status.

## Scope boundaries

- The supported target is exactly two GB10/SM121 systems with TP=2. Root-level
  TP4 files are inherited experimental material, not a supported product path.
- The default loads the vision tower because the historical video appliance did,
  but the video workload itself was text-only. Multimodal capacity and quality
  are not certified by that text throughput result. `LANGUAGE_MODEL_ONLY=1` is
  the lower-memory custom alternative.
- DFlash2 is the published-video default and is separately licensed CC
  BY-NC-ND 4.0. The default is therefore not appropriate for commercial use.
  It may also perform very differently across output styles because speculative
  acceptance varies; it is not a universal speed multiplier. Select MTP or
  no speculation when the license or workload does not fit.
- The primary synthetic isolation fixture is a reproducible service-stress
  proxy, not a claim that repetitive filler represents every agent, tool,
  reasoning, or multimodal workload.
- The API binds to loopback by default. A non-loopback bind requires a bearer
  key unless the operator explicitly accepts unauthenticated LAN exposure.
  Readiness/warmup probes currently use loopback; use an SSH tunnel or an
  authenticated `API_HOST=0.0.0.0` bind rather than a LAN-only bind.

## Distribution and development

- The source preview has no qualified prebuilt registry image. First use
  builds native code; see `docs/IMAGE_RELEASE.md` for the binary-publication gate.
- Candidate builds are explicitly unqualified and use separate declared
  hashes rather than modifying the frozen video reference. New/deleted
  inventory entries need tooling review; see `docs/CANDIDATE_BUILDS.md`.
- The launcher retains Mia's default container names. Follow the migration
  section in `SPARKGLM.md`; do not use one recipe's stop/restart command while
  assuming the other recipe's same-named containers are independent.

## Research archive

`research/` intentionally contains incomplete and rejected work. Those files
are useful provenance and negative evidence, not alternate supported install
paths. Atlas code remains AGPL-3.0-only and never becomes Apache merely because
it is stored in this repository.
