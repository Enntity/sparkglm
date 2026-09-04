# Final follow-up experiments

These files were uncommitted research in the private working tree when the
publication candidate was assembled.

- `rejected-prefill-fusion-experiments.md` records the measured outcome.
- `uncommitted-experiments.patch` preserves the exact candidate changes.
- `Dockerfile.route-profile` and the Python tools support route-locality and
  expert-promotion analysis.

The fused-activation variants were rejected because full-model results did not
justify their complexity and C1 regression. The route profiler intentionally
synchronizes the host and must never be enabled in production.
