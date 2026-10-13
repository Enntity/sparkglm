# SparkGLM maintenance build — 2026-10-13

The image includes per-node fabric configuration and Atlas commit `6f27e8fa`, which preserves the full blocking assistant content alongside refusal metadata for Atlas-Inf/atlas#172. The classifier, tool gating, DFlash, profile defaults, and display-carveout placement are unchanged.

The native arm64 CUDA image built successfully on GB10 with all configured build-time regression gates, including 25 refusal and 43 streaming tests. These tests exercise production response assembly and serialization; they are not model inference benchmarks. Product checks also passed: 27 installer/fabric tests, 3 converter tests, shell checks, Python syntax and historical receipt hashes.

The published image and source identities are in `qualification.json`. The build preserved the resident model containers. Live GLM inference qualification awaits a coordinated window on the pair, which is reserved for an MTP experiment. Historical RC4 performance results remain historical.
