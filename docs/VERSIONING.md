<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Product versions

SparkGLM and SparkQwen use independent semantic versions, starting at `0.1.0`.
The number identifies the distribution, including its Atlas pin, launch recipe
and profiles. It is separate from the model's name and the Atlas engine version.
Historical dated releases and release-candidate numbers remain available.

`install/release.json` is the source of truth. `./start.sh --version` prints the
checkout version and exact pinned engine without needing Docker, weights, `.env`
or a worker. `./start.sh status` also reads the version label and image ID from
each deployed container, so an older deployment is distinguishable from a newer
checkout. Older images without the label may show an empty version; use their image ID.

For an image already installed locally, `docker run --rm IMAGE --version` prints
its product version and engine pin without loading the model. Container startup
logs include the same fields. OCI `org.opencontainers.image.version` records the
product version, and the existing install-tree label records its exact recipe.

Before publishing a changed distribution, bump the version in `install/release.json`:

- Patch: compatible fixes to the engine, recipe or packaging.
- Minor: new functionality; during `0.x`, also changes requiring user action.
- Major: incompatible changes once the product reaches `1.0.0`.

Documentation-only changes do not need a version bump. Do not reuse a published
version for different distribution contents. Tag GitHub releases `vMAJOR.MINOR.PATCH`
and record the product commit, full install tree, Atlas commit/tree and image digest
in the release assets. Engine pins and image digests remain immutable identities;
versions do not replace them. Image tags remain the first 12 characters of the
`install/` tree hash, and `start.sh` continues to select that exact tag.

## Benchmark each release

Before promoting each version, serve its candidate image on two DGX Sparks in Spark mode and run the full pinned SparkModelBench suite through `bench/run.sh`. Use the default lanes and keep failed quality checks and incomplete streams visible. Run the existing correctness and topology checks as well. [Benchmark setup](REPRODUCE.md#current-suite)

Archive the complete run and its checksums privately. Copy the reviewed scorecard to `results/sparkmodelbench-v<VERSION>/CARD.md`. Give it the heading `# SparkGLM v<VERSION> · SparkModelBench`, identify the measured suite revision, and keep the four result tables. Retain public model and TP details and remove private appliance fields. Include the public files in that directory’s `SHA256SUMS` and link the scorecard in the release notes. CI checks that the version in `install/release.json` has its own four-lane scorecard.
