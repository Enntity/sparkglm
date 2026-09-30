# Licensing

SparkGLM is AGPL-3.0-only ([LICENSE](../LICENSE)). Files that came from
elsewhere keep their own license, which their SPDX header names. Ideas we took
from other projects, without their code, are credited in the README's
[Credits](../README.md#credits).

## In this repository

| Material | License |
|---|---|
| `start.sh`, `install/`, `bench/`, docs and results | AGPL-3.0-only |
| `install/converter/` NVFP4 quantization kernel | AGPL-3.0-only, copied unchanged from [Atlas-Inf/atlas](https://github.com/Atlas-Inf/atlas) `kernels/gb10/common/quantize_bf16_to_nvfp4.cu` by way of Mango-kid/atlas ([`provenance.json`](../install/converter/provenance.json)) |
| `install/flash_kda/flash_kda_sm121_slots.patch` | MIT (FlashKDA context, [`LICENSES/MIT-FlashKDA.txt`](../LICENSES/MIT-FlashKDA.txt)) plus AGPL-3.0-only changes |
| `install/flash_kda/atlas_flash_kda_bridge.cu` | AGPL-3.0-only, with its workspace-size and launch arithmetic adapted from FlashKDA `csrc/flash_kda.cpp` (MIT, [`LICENSES/MIT-FlashKDA.txt`](../LICENSES/MIT-FlashKDA.txt)) |
| `install/start-node.sh`, `install/convert.sh` | AGPL-3.0-only; the container flags and the converter's verify pass follow [LLooM](https://github.com/Enntity/lloom)'s Atlas recipe (MIT, [`LICENSES/MIT-LLooM.txt`](../LICENSES/MIT-LLooM.txt)) |
| The `mia-*` prompts in `bench/staggered_openai.py`, and `.github/FUNDING.yml` | MIT, Copyright (c) 2026 Mia's AI Lab ([`LICENSES/MIT-Mia.txt`](../LICENSES/MIT-Mia.txt)), quoted from MiaAI-Lab's recipe while it was MIT |
| The filler sentence and three decode prompts in `bench/mmastrac_*.py` | quoted verbatim from [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10), which has no license. They are short fixed strings kept for comparability; the code around them is ours |

The harness copies under `results/*/raw/harness/` are unedited receipts: they
carry the same prompts as the `bench/` files above, under the same terms. The
RigMark receipts in `results/` are output of RigMark (MIT, Alex Ellis /
OpenFaaS Ltd).

## Fetched while building the image

| Component | License |
|---|---|
| Atlas engine, [`Enntity/atlas`](https://github.com/Enntity/atlas) at the pinned commit | AGPL-3.0-only |
| FlashKDA (MoonshotAI) | MIT |
| FlashInfer | Apache-2.0 ([`LICENSES/Apache-2.0.txt`](../LICENSES/Apache-2.0.txt)) |
| NVIDIA's sparse-MLA SM120 prefill source inside FlashInfer (`csrc/sparse_mla_sm120_prefill.cu`, `include/flashinfer/attention/sparse_mla_sm120/`) | BSD-3-Clause, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES ([`LICENSES/BSD-3-Clause-NVIDIA.txt`](../LICENSES/BSD-3-Clause-NVIDIA.txt)) |
| CUTLASS | BSD-3-Clause |
| The engine's Rust dependencies (`Cargo.lock`) | mostly MIT, Apache-2.0 or BSD |
| Other models' chat templates in the engine's `jinja-templates/`, copied into the image but unused by GLM | their model vendors' terms |
| CUDA base images and Ubuntu packages | their own terms |

The engine's GLM-5.3-Flash support started as Reiner Schmidt's port
([Mango-kid/atlas](https://github.com/Mango-kid/atlas/tree/feat/glm53-dual-spark),
AGPL-3.0-only); see the engine's `docs/porting/GLM_5_3_FLASH.md`. The
confidence-width option's starting calibration table comes from knapcio's
[GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
(MIT, Copyright (c) 2026 knapcio). Parts of the engine's sparse-MLA prefill
preparation were drafted by supervised DeepSeek model workers and then reviewed
by us.

The image ships the FlashInfer, NVIDIA, FlashKDA and CUTLASS notices under
`/opt/atlas/notices/` and the engine's license at `/LICENSE`. The license,
copying and notice files of the third-party Rust crates compiled into the
engine are under `/opt/atlas/notices/rust/` (`INDEX.tsv` lists each crate, its
declared license and its source; `install/rust-notices.py` collects them at
build time). All of them are under permissive licenses (MIT, Apache-2.0, BSD,
ISC, Zlib, Unicode-3.0, CDLA-Permissive-2.0 or dual/multi-licensed with one of
these). Images built before this change do not carry that directory. Its
complete corresponding source is:

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
