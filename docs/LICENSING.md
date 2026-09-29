# Licensing

SparkGLM is AGPL-3.0-only ([LICENSE](../LICENSE)). Files that came from
elsewhere keep their own license, which their SPDX header names.

## In this repository

| Material | License |
|---|---|
| `start.sh`, `install/`, `bench/`, docs and results | AGPL-3.0-only |
| `install/converter/` NVFP4 quantization kernel | AGPL-3.0-only, copied unchanged from Mango-kid/atlas ([`provenance.json`](../install/converter/provenance.json)) |
| `install/flash_kda/flash_kda_sm121_slots.patch` | MIT (FlashKDA context, [`LICENSES/MIT-FlashKDA.txt`](../LICENSES/MIT-FlashKDA.txt)) plus AGPL-3.0-only changes |
| `bench/four_stream_video.py` | Apache-2.0, vLLM contributors ([`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt)) |

`.github/FUNDING.yml` keeps the MiaAI-Lab sponsor link from the recipe
SparkGLM grew out of.

## Fetched while building the image

| Component | License |
|---|---|
| Atlas engine, [`Enntity/atlas`](https://github.com/Enntity/atlas) at the pinned commit | AGPL-3.0-only |
| FlashKDA (MoonshotAI) | MIT |
| FlashInfer, including NVIDIA's sparse-MLA prefill source | Apache-2.0 |
| CUTLASS | BSD-3-Clause |
| CUDA base images and Ubuntu packages | their own terms |

The image ships these notices under `/opt/atlas/notices/` and the engine's
license at `/LICENSE`. Its complete corresponding source is:

- this repository at the commit whose `install/` tree matches the image tag;
- the Atlas commit recorded in `/opt/atlas/source-manifest.json`;
- the upstream revisions pinned in `install/`.

If you modify the engine and let others use it over a network, AGPL section 13
requires you to offer them your modified source.

## Models

No weights are distributed here. `./start.sh` downloads them from their
publishers:

- [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4): MIT.
- [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2):
  **CC BY-NC-ND 4.0, non-commercial.** Both shipped profiles use this drafter.

Read each model card before use. The converted overlay is a local derivative
of the NVIDIA checkpoint and stays on your machines.

## Earlier versions

The vLLM-era tree carried more components under more licenses: Mia's MIT
recipe, vLLM Apache-2.0 backports, ExLlamaV3 MIT code, the ShapleyMCG-licensed
EXL3 checkpoints, and the GLM-5.3 chat template. That tree, with all its
notices, is preserved at the tag `vllm-final`.
