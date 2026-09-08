# Local preparation receipt — 2026-09-06

State: prepared, unqualified NVFP4 experiment. No performance claim.

- `python3 tests/test_nvfp4_candidate.py`: six tests pass. They check copied
  file hashes and pins, Python/shell syntax, actual head/worker argument
  generation through a recording executable, immutable download plans,
  checkpoint config/shard rejection, and benchmark command calibration.
- `python3 research/experiments/nvfp4-tp2/check_source.py --vllm-source ...`:
  passes against vLLM `487ecf187d3dfe74d2cf6119a92881dba403c219` with the
  existing SparkGLM DFlash2 installer applied first. The copied MXFP8 installer
  applies successfully and its second application is byte-idempotent.
- `scripts/check.sh all`: passes G0/publication checks, including all existing
  qualification records, licensing, attribution and publication privacy.
- Public GHCR base manifest resolved to the digest in `candidate.json`.
  Public Hugging Face configurations resolved and compared; the target's
  current compressed-tensors metadata differs from the pinned August 28
  ModelOpt metadata. Candidate pins the latter explicitly.

No SSH or endpoint requests to the Sparks were made. No weights or image
layers were downloaded. The local Docker daemon was unavailable; the source
derivative has not been built. Patch compatibility is a source test, not proof
of the complete image contents or GPU execution. G1 through G5 are unrun.

Outstanding hardware work includes Humming operator/graph validation, NVFP4
tinyGLM integration, full checkpoint loading, multilingual/tool corruption
checks, and matched mixed-prefill measurements. EXL3 tinyGLM receipts and
upstream benchmarks do not qualify this candidate. All original defaults and
the running fleet remain untouched.
