# Contributing

SparkGLM accepts narrowly scoped work for GLM-5.3-Flash on two DGX Spark GB10
systems.

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

Every optimization commit should use the provenance structure documented in
`docs/PROVENANCE.md`.
