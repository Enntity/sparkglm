# Research archive

Nothing under this tree is a second supported distribution. Root-level
Dockerfiles whose names include `binary` or `derivative`, plus `start-tp4.sh`,
are likewise historical qualification/unsupported research helpers; only the
root `Dockerfile` and `start.sh` define the current two-Spark path.

This directory preserves work that informed the current engine but is not the
recommended serving path.

- `current-engine-history/` preserves the accepted clean patch series and
  commit attribution without importing unsafe historical git objects.
- `atlas/` contains the native Rust/CUDA attempt as an AGPL-preserving patch,
  its GLM-specific probes, and its status reports.
- `vllm-iterations/` contains the earlier source-fork campaign, including raw
  results and negative experiments.

Nothing under this directory is enabled merely by being present. Read the
nearest README and result record before applying a patch.
