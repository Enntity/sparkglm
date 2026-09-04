# Attribution and provenance

This file records the direct external lineage of the publication candidate.
It distinguishes a dependency, a direct port, an adaptation, inspiration, and
original SparkGLM work. It is not a substitute for the license text retained
in `LICENSES/` or file-level SPDX identifiers.

## Runnable vLLM/EXL3 engine

| Source | Exact revision | License | Relationship |
| --- | --- | --- | --- |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | `eb0469fbb2b49fd7c025f594a3339a121e58f7a9` | MIT | Serving-recipe, GLM integration, E2 prefill, DFlash2, cache, scheduler-control, warmup, and two-node base |
| [zai-org/GLM-5.3 chat template](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/chat_template.jinja) | `aca966e4e02791568aa6a4ced368624b3d897f42` | GLM-5.3 License | Chat serialization base; imported `None` guard and early-exit correctness/performance update for tool-result reordering, with SparkGLM's inherited reasoning and multimodal extensions retained |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | `487ecf187d3dfe74d2cf6119a92881dba403c219` | Apache-2.0 | Runtime source base used by the reproducible build |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | `c5d9c657966ffeeaa9353f0cc899f18629da4a13` | MIT | EXL3 K4 trellis decode, Hadamard helpers, and tensor-core MMA foundations |
| [Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) | `0c03250cd7176a2fef9cbbf9329fed08c8750e7d` | Apache-2.0 with retained Mia MIT notice | M64 `cp.async` fat-GEMM pipeline adapted into the current kernel |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) | `67dc742` | Apache-2.0 | Cooperative-grid decode direction; wrapper and kernel were not copied |
| [vLLM PR 53109](https://github.com/vllm-project/vllm/pull/53109) | `515c2470db4b` | Apache-2.0 | Packed fused-RMSNorm output fix, directly ported |
| [vLLM PR 52696](https://github.com/vllm-project/vllm/pull/52696) | `2908700406` campaign tip | Apache-2.0 | Native FP16 sparse-selector capability, ported; FP16 remains off by default in the accepted image |
| [vLLM PR 52805](https://github.com/vllm-project/vllm/pull/52805) | `12f64b39d29282437e35be9aa5db432fb2a1a6e6` | Apache-2.0 | XGrammar termination correctness backport inherited from Mia |
| [vLLM PR 53046](https://github.com/vllm-project/vllm/pull/53046) | `c6e19b3be24338759a443e03c8325d76da9ee202` | Apache-2.0 | Speculative-reasoning FSM correctness backport inherited from Mia |

### Original SparkGLM boundaries

The following are original project work built on the foundations above:

- The GPU-resident grouped-prefill task planner and phase-wide execution
  contract across fat experts.
- Persistent route scratch, cross-expert parallel scheduling, FP32 atomic
  routed accumulation, exact GLM-5.3/TP2 shape guards, and rollback wiring.
- The TP2-local cooperative whole-layer decode contract, expert-sorted
  multi-row tiling, graph-safe persistent buffers, bounded dispatch, and exact
  GLM clamp/SwiGLU integration.
- tinyGLM, deterministic model-free and endpoint A/B gates, staggered workload
  harnesses, capacity admission checks, and publication-oriented evidence.
- The merge and regression coverage that keeps Z.ai's pinned chat-template
  behavior together with the recipe's reasoning toggle and multimodal sentinels.

SparkGLM did not invent EXL3 arithmetic, vLLM scheduling, DFlash2, GLM-5.3,
the M64 `cp.async` tile, packed RMSNorm, or the FP16 sparse selector.

## Model artifacts not redistributed here

| Artifact | Author/source | License boundary |
| --- | --- | --- |
| GLM-5.3-Flash | [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) | Model license at the source repository |
| EXL3/TR3 4bpw weights | [brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | ShapleyMCG License 1.0 at the source repository |
| DFlash2 drafter | [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | CC BY-NC-ND 4.0 at the source repository |

No checkpoint or derived direction tensor from these projects is committed.

## Atlas-native archive

| Source | Exact revision | License | Relationship |
| --- | --- | --- | --- |
| [Atlas-Inf/atlas](https://github.com/Atlas-Inf/atlas) | `bdcccc2ca91eba084aac94a059e3b0f4a5d556dd` | AGPL-3.0-only | Native Rust/CUDA engine base |
| SparkGLM Atlas archive | `775cb3655e29a3735f4f58faa540608f9427bf51` | AGPL-3.0-only | GLM parser, typed state, KDA/DSA/MoE/EXL3 work, probes, and incomplete end-to-end integration |
| [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) | `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` | MIT | KDA prefill kernel source used by the Atlas experiment |
| NVIDIA CUTLASS | `5c149f52a436782210263fb2f19b354443a61c6a` | upstream license | FlashKDA build dependency pin |

The Atlas patch and copied Atlas-derived documentation remain explicitly
AGPL-3.0-only. No attempt is made to relicense Atlas work as Apache or MIT.

## Contributor handles

Direct public credits relevant to the current engine include:

- MiaAI-Lab: `@MiaAI_lab`
- plotarmordev: `@plotarmordev`
- turboderp: `@turboderp_`
- Artem Matskevych / Reederey87: `@Reederey`
- Victor Cruz: `@ViC305`
- Woosuk Kwon: `@woosuk_k`
- Roy Wang: `@esmeetu87`
- vLLM: `@vllm_project`
- Brandon: `@BrandonMusicKy`
- Inco AI: `@inco_ai`
- Z.ai: `@Zai_org`

The inherited XGrammar correctness work also includes Flora Feng (`sfeng33`),
Chauncey Jiang (`chaunceyjiang`), and Zbigniew Majewski (`knapcio`). Public X
handles were not verified for those contributors, so GitHub identities are the
authoritative attribution.
