# Optimization provenance policy

Every new commit that ports, adapts, or is materially inspired by external work
must make the lineage visible in the commit message. Source comments and design
documents are useful, but they do not replace commit-level attribution.

The preserved patch mailboxes predate this release policy and remain immutable
historical evidence. We do not rewrite their messages; their lineage is
normalized in `provenance/upstreams.json` and `docs/ATTRIBUTION.md`. CI applies
this policy to every new pull-request commit that touches runtime paths.

Use this body structure:

```text
Provenance:
- <project URL and exact revision>: <what was reused or inspired the work>

Original work:
- <what this commit designed or implemented specifically for this repository>

Verification:
- <tests and measured result>
```

Rules:

- Name the upstream project, URL, and exact commit whenever one is known.
- Say whether code was copied, adapted, ported, or merely inspired by it.
- Preserve upstream copyright and license notices in copied or derived files.
- Do not use `Co-authored-by` unless that person actually authored part of the
  commit; provenance belongs in explicit attribution instead.
- If the implementation is original but uses standard primitives or published
  ideas, identify those foundations and state the original boundary precisely.
- If there is no external lineage, write `Provenance: original implementation`
  rather than leaving the question ambiguous.
- `scripts/check_commit_provenance.py BASE_SHA HEAD_SHA` enforces the three
  sections. External provenance must include a canonical HTTPS URL and full
  40-character revision.

## Current EXL3 cooperative-decode lineage

- [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) at
  `c5d9c657966ffeeaa9353f0cc899f18629da4a13`: EXL3 K4 trellis decoding,
  Hadamard helpers, and tensor-core MMA arrangement.
- [`vcruz305/vllm-exl3`](https://github.com/vcruz305/vllm-exl3) at
  `67dc7426dfbdecbc1527199eb32c0d328d8f609f`:
  inspiration for applying a cooperative
  CUDA grid to small-batch EXL3 MoE decode.
- This repository: the TP2-local GLM-5.3 shape, whole-layer launch contract,
  expert-sorted multi-row tiling, persistent scratch, GLM clamp/SwiGLU
  semantics, FP32 routed accumulation, graph-safe integration, bounded
  dispatch, and the tinyGLM A/B gate.

## GPU-resident grouped-prefill lineage

- [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) at
  `c5d9c657966ffeeaa9353f0cc899f18629da4a13`: EXL3 K4 trellis decoding,
  Hadamard helpers, and tensor-core MMA arrangement.
- [`Reederey87/glm53-flash-exl3-2x-dgx-spark`](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)
  at `0c03250`: the M64
  `cp.async` fat-GEMM pipeline adapted by this repository's existing
  `exl3_fat_gemm_kernel` implementation.
- This repository: the device-side M64 task planner, persistent route scratch,
  phase-wide gather/gate-up/activation/down scheduling, cross-expert parallel
  execution, FP32 atomic routed accumulation, exact GLM-5.3 TP2 shape guard,
  Python dispatch, and model-free plus tinyGLM comparison gates.

No grouped-prefill implementation was copied from either upstream project.
The new grouped orchestration is original work which calls arithmetic derived
from the existing attributed fat-GEMM kernel.
