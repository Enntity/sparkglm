# Contributing

SparkGLM accepts narrowly scoped work for GLM-5.3-Flash on two DGX Spark GB10
systems.

Start with [docs/METHODOLOGY.md](docs/METHODOLOGY.md). Every change must name
its state, target workload and metric, protected behavior, applicable gate
level, and rollback. Use the pull-request template; do not substitute an
isolated peak result for its requested evidence.

Read [docs/LICENSING.md](docs/LICENSING.md) before carrying code, templates,
weights, or results across project boundaries. Update
`provenance/upstreams.json` whenever an upstream, revision, relationship,
notice, or affected path changes.

- Preserve exact upstream attribution and license notices.
- State whether external work was copied, adapted, ported, or merely inspired
  the change.
- Put one logical optimization in each commit.
- Include correctness evidence before performance evidence.
- Report exact source revisions, model and quantization, launch configuration,
  warmup, prompts, arrival pattern, and raw results.
- Compare equivalent quality and precision. A smaller checkpoint is not a free
  performance win.
- Retain negative results when they prevent others from repeating an expensive
  experiment.
- Do not commit weights, tensors, compiled binaries, credentials, machine-local
  paths, `.env` files, or generated videos.
- Run `scripts/check.sh all` before every pull request.
- For runtime changes, also run `scripts/check.sh container` in the built image
  and the applicable tinyGLM/full-model gates; `all` deliberately means all
  public-runner-safe checks, not access to private hardware.
- Add a result bundle for every performance claim or promoted default; validate
  it with `python3 scripts/qualification.py verify-all`.
- Keep a candidate behind a rollback flag until its required full-model and
  semantic gates pass.
- Never run untrusted public pull-request code automatically on private Spark
  hardware. Maintainers qualify reviewed commits.

Every optimization commit should use the provenance structure documented in
`docs/PROVENANCE.md`. Pull-request CI checks runtime commit messages with
`scripts/check_commit_provenance.py`; do not squash away distinctions among
separate optimizations before review.

Passing tinyGLM is required for relevant engine changes but is not a
full-model speed or quality claim. See [docs/QUALIFICATION.md](docs/QUALIFICATION.md)
for the result format and creation command.
