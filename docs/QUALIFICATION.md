# Qualification records

A SparkGLM qualification is a small directory containing:

```text
results/<state>/<experiment-id>/
  qualification.json
  RESULT.md                 # human report, when available
  raw/                      # raw JSON/log evidence, when available
```

`qualification.json` is authoritative for identity and scope. Human reports
explain the result; raw artifacts support it. Every artifact named by the
record carries a SHA-256 checksum.

Validate all checked-in records with:

```bash
python3 scripts/qualification.py verify-all
```

`checksum` discovers newly added bundle files, assigns a conservative artifact
kind, and binds every file to the record. Review those inferred kinds before
attestation.

For the frozen full-model workload, use the same pair ID for both arms and
alternate which arm is run first across repeated campaigns:

```bash
export SPARKGLM_PAIR_ID=2026-09-04-my-experiment
export SPARKGLM_IMAGE_DIGESTS=sha256:<rank0>,sha256:<rank1>
export SPARKGLM_MODEL_REVISION=<immutable-model-revision>
# Run one repetition at a time so retained first-arm order can alternate.
export SPARKGLM_FULL_REPETITION_ID=1
scripts/full-model-gate.sh baseline \
  results/candidates/2026-09-04-my-experiment <served-model-name>
# restart at the candidate commit with otherwise identical configuration
scripts/full-model-gate.sh candidate \
  results/candidates/2026-09-04-my-experiment <served-model-name>
```

The collector writes C1 16K/32K, staggered C2 16K/32K, and staggered C4 16K
receipts with exactly calibrated prompts and 400 delivered tokens per request.
It also captures the served-model response, source revision, declared rank
image digests, and non-secret configuration. Omit
`SPARKGLM_FULL_REPETITION_ID` only when collecting a single arm's three
repetitions in one launch; a publishable comparison should run one repetition
at a time and alternate which arm runs first. Record the six-arm order in
`environment.run_order`.

Create a new candidate skeleton from a clean candidate commit with:

```bash
python3 scripts/qualification.py new \
  --id 2026-09-04-my-experiment \
  --title "My experiment" \
  --target "32K staggered C2 TTFT" \
  --baseline-ref <baseline-commit>
```

The command captures the candidate Git commit and whether the worktree is
clean. Put raw outputs inside the new bundle, fill in the frozen configuration,
metrics, gates, limitations, and attestation, then run:

```bash
python3 scripts/qualification.py checksum \
  results/candidates/2026-09-04-my-experiment/qualification.json
python3 scripts/qualification.py verify-all
```

## Qualification levels

- `legacy`: migrated evidence that predates this schema; useful but never
  retroactively certified;
- `G0` through `G5`: the highest completed gate from `docs/METHODOLOGY.md`.

The level describes what was proved, not how exciting the number is. A G1
kernel win cannot be presented as an endpoint win. A G2 tinyGLM result cannot
be presented as full-model throughput. A G3 performance result without G4
must preserve its explicit quality limitation.

## Attestation

`attestation.status` is one of:

- `unattested`: ordinary experiment or candidate;
- `legacy`: migrated before the qualification policy;
- `maintainer-reviewed`: rerun or verified by a SparkGLM maintainer at the
  exact candidate commit.

Only a clean commit with all gates through its declared level passing may be
`maintainer-reviewed`. The Git commit containing the qualification should be
signed when it supports a public headline or release. This is an internal
project qualification, not third-party hardware certification.

## Historical migrations

Historical reports and raw receipts are retained under `results/legacy/`.
Their manifests state what is missing. Migration does not upgrade old evidence
or invent commit, image, topology, warmup, or quality facts that were not
recorded at the time.

## Status and location

New work uses:

- `results/candidates/` while testing;
- `results/accepted/` after a successful scoped qualification;
- `results/rejected/` for useful negative results;
- `results/legacy/` for pre-policy evidence.

Moving a directory does not itself promote a change. Update the manifest,
complete the required gates, obtain maintainer review, and commit the evidence.
