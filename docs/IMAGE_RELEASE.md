# Prebuilt image status

The public-source preview uses a local source build. No qualified SparkGLM
registry image is currently offered. Do not replace the image with an upstream
tag and claim it reproduces the SparkGLM engine.

A tested ARM64 image is retained privately, but performance testing does not
clear a binary for redistribution. Publishing it is a separate deliverable,
not a prerequisite to reading, building, or contributing to this source tree.

Before offering an image, maintainers must:

1. Inventory every image layer and installed package for weights, credentials,
   private configuration, caches, and excluded artifacts. Inspect deleted
   content in lower layers too, not just the final filesystem.
2. Check component redistribution terms and preserve required notices and
   corresponding-source obligations, including the pinned base image. A root
   Apache license is not blanket permission for the whole container.
3. Publish a component/SBOM and source-revision manifest, exact platform and
   registry digest, and the applicable license notices. Never put target or
   DFlash2 weights in the image.
4. Pull the exact registry digest onto a clean pair, verify rank identities,
   run the appropriate integration/serving gates, and record the evidence.
5. Provide a digest-pinned launcher path that verifies the recipe identity;
   a mutable tag or SKIP_BUILD escape hatch is not a release mechanism.

Until then, use [the source quickstart](../SPARKGLM.md). This status is explicit
so a public source preview is not mistaken for a G5-certified appliance or an
already audited binary distribution.
