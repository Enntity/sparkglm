# Working in this repository

- Keep it simple: one entry point (`start.sh`), one install directory, one
  results tree. Prefer removing code to adding it.
- The image tag is the git tree of `install/`. Anything that changes the image
  must live in `install/`, and nothing else should.
- Engine changes belong in `Enntity/atlas` on `spark-shared`, the common
  engine used by SparkGLM and SparkQwen. Keep model-specific execution gated
  by model type and kernel target. Open PRs only in Enntity repositories.
  After qualification, update the exact commit and tree together in
  `install/atlas-source.json` and document the tested product image. Do not
  reintroduce a separate active SparkGLM engine branch.
- A performance claim needs raw receipts and `SHA256SUMS` under `results/`,
  with its baseline measured on the same pair. Report every repetition.
- Keep lossy or quality-affecting optimizations off by default.
- New files carry an SPDX header. Keep third-party headers and licenses intact
  (see `docs/LICENSING.md`).
- Before sending a change, run the checks in `.github/workflows/static.yml`.
