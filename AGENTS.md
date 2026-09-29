# Working in this repository

- Keep it simple: one entry point (`start.sh`), one install directory, one
  results tree. Prefer removing code to adding it.
- The image tag is the git tree of `install/`. Anything that changes the image
  must live in `install/`, and nothing else should.
- Engine changes belong in `Enntity/atlas`: upstream-worthy work on
  `upstream/glm53-flash`, and SparkGLM-only work on the `sparkglm/*` branch.
  Then update the pin in `install/atlas-source.json`.
- A performance claim needs raw receipts and `SHA256SUMS` under `results/`,
  with its baseline measured on the same pair. Report every repetition.
- Keep lossy or quality-affecting optimizations off by default.
- New files carry an SPDX header. Keep third-party headers and licenses intact
  (see `docs/LICENSING.md`).
- Before sending a change, run the checks in `.github/workflows/static.yml`.
