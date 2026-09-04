# SparkGLM repository instructions

## Provenance

For every optimization commit, follow `docs/PROVENANCE.md` and
`docs/LICENSING.md`. The commit message
must distinguish copied, adapted, inspired, and original work; name upstream
URLs and exact revisions when known; and record verification. Preserve all
applicable upstream copyright and license notices in source files.
Keep `provenance/upstreams.json` synchronized with every external source,
revision, relationship, license/notice file, and affected path.

## Publication safety

- Run `scripts/publication-audit.sh` before every release-oriented commit.
- Never add weights, model-derived tensors, compiled GPU binaries, credentials,
  machine-local `.env` files, private hostnames, or generated videos.
- Keep Atlas-derived material under `research/atlas/` and AGPL-3.0-only.
- Keep rejected experiments labeled as rejected; their presence is not a
  recommendation to enable them.
- Do not change repository visibility or publish releases without explicit
  user approval after the private review checklist is complete.

## Change qualification

- Read `docs/METHODOLOGY.md` before modifying performance-sensitive code.
- Run `scripts/check.sh all` for every change.
- Keep experiments disabled until they pass the gate required by their change
  class. tinyGLM is a required integration gate, not production evidence.
- Put every performance claim, accepted optimization, or rejected experiment
  in a checksum-bound result bundle under `results/`.
- Do not call a change promoted or alter the recommended default without a
  maintainer-reviewed qualification at the required level.
- Do not execute untrusted pull-request code on private Spark hardware.
