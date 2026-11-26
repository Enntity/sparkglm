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
- Qualify every new product version with the full pinned SparkModelBench suite
  through `bench/run.sh`. Publish its reviewed scorecard as described in
  `docs/VERSIONING.md`. Keep raw run logs and private appliance details local.
  Measure performance comparisons on the same pair and retain every repetition.
- Keep lossy or quality-affecting optimizations off by default.
- New files carry an SPDX header. Keep third-party headers and licenses intact
  (see `docs/LICENSING.md`).
- Before sending a change, run the checks in `.github/workflows/static.yml`.
