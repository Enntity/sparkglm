# SparkGLM repository instructions

## Provenance

For every optimization commit, follow `docs/PROVENANCE.md`. The commit message
must distinguish copied, adapted, inspired, and original work; name upstream
URLs and exact revisions when known; and record verification. Preserve all
applicable upstream copyright and license notices in source files.

## Publication safety

- Run `scripts/publication-audit.sh` before every release-oriented commit.
- Never add weights, model-derived tensors, compiled GPU binaries, credentials,
  machine-local `.env` files, private hostnames, or generated videos.
- Keep Atlas-derived material under `research/atlas/` and AGPL-3.0-only.
- Keep rejected experiments labeled as rejected; their presence is not a
  recommendation to enable them.
- Do not change repository visibility or publish releases without explicit
  user approval after the private review checklist is complete.
