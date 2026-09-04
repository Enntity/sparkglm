# Pinned upstreams

## vLLM engine

- Repository: `https://github.com/vllm-project/vllm.git`
- Commit: `487ecf187d3dfe74d2cf6119a92881dba403c219`
- Version reported by the installed engine: `0.1.dev20051+g487ecf187`
- License: Apache-2.0

This is the exact engine revision named by Mia's current two-Spark recipe. The
active SparkGLM branch is based directly on this commit rather than copying an
installed Python tree or container filesystem.

## Mia two-Spark recipe

- Repository:
  `https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git`
- Commit: `c707598ebcf02fd827d079a7c47e785069425efe`
- License: MIT
- Snapshot location: `recipes/mia/`

This pin includes the E2 fat-expert prefill kernel, 7,168-token prefill default,
DFlash draft TP2 default, K-pool tail correction, optional indexer-workspace
right-sizing, and numeric spin-wait configuration available on 2026-09-01.

## Reederey two-Spark production reference

- Repository:
  `https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark.git`
- Commit: `b229968a64ae3a270acdda9ce539a421e21598d7`
- License: Apache-2.0 for original work; MIT for identified vendored recipe files
- Snapshot location: `recipes/reederey/`

This is a measured production evolution of the Mia recipe, not a replacement
engine. It is pinned because it adds useful evidence and implementations around
mixed-prefill aging, per-group prefix-cache retention, fine-grained cache hits,
memory floors, and DFlash topology. Its reported results are external evidence
until they reproduce on our pair and workload.

## Model artifacts

- Target checkpoint: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- Model revision: `25a44fdbf16862a46b7cc9921142c6c81350af2f`
- DFlash2 checkpoint: `incoai/GLM-5.3-Flash-DFlash2`

Model files and credentials are not committed to this repository.

## Preserved Atlas work

The private `Enntity/sparkglm` repository contains two independent recovery
points created before this branch:

- branch and tag `archive/atlas-native-final-2026-09-01` /
  `atlas-native-final-2026-09-01`, ending at commit `c1d4b17`;
- branch and tag `archive/atlas-source-complete-2026-09-01` /
  `atlas-source-complete-2026-09-01`, a self-contained snapshot of the larger
  271-file Atlas working tree.

The missing upstream `assets/atlas-demo.mp4` LFS object was deliberately omitted
from the self-contained snapshot. It is unrelated to GLM source or results.
