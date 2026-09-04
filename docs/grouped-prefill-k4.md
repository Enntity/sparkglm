# GPU-resident grouped EXL3 prefill

## Problem

The previous GLM-5.3 fat-expert path copied the 288 expert counts to the CPU,
waited for that copy, then issued gather, gate/up, activation, and down kernels
one expert at a time. At 16K-32K prefill, many experts overflow the 128-row
fused-MoE scratch. The arithmetic kernels are useful, but the host-driven
execution shape serializes independent experts and pays dozens to hundreds of
launches per layer.

## Candidate

`EXL3_GROUPED_PREFILL_K4=1` enables the promoted exact-shape path for K4 MCG,
hidden-size 4096, and TP-local intermediate-size 1024. A one-thread GPU planner
turns device-resident expert offsets into compact M64 work items. Five
phase-wide launches then execute gather/Hadamard, paired gate+up GEMM,
clamped SwiGLU/down Hadamard, and down GEMM/scatter across all fat experts.
Thin experts continue through the existing fused kernel. The host never reads
expert counts. Down projections use FP32 atomic accumulation because experts
now overlap.

The first prototype used one cooperative launch with grid barriers. It was
3-13% slower because its expert loop serialized the GPU. It was discarded.

## Controlled evidence

On one GB10 with production TP-local dimensions, the model-free A/B compared
the prior per-expert pipeline against the grouped pipeline and got bit-for-bit
identical output (`max_abs=0`) in every measured shape:

| Tokens | Fat experts x rows | Old ms | Grouped ms | Speedup |
|---:|---:|---:|---:|---:|
| 512 | 16 x 256 | 3.133 | 2.062 | 1.519x |
| 1024 | 8 x 512 | - | - | 1.273x |
| 1024 | 10 x 512 | - | - | 1.261x |
| 1024 | 10 x 920 | - | - | 1.123x |
| 1024 | 4 x 1024 | - | - | 1.095x |
| 1024 | 12 x 1024 | - | - | 1.119x |

Those numbers include the old GPU launch sequence but exclude its CPU count
readback, so they do not credit the candidate for removing that synchronization.

The deterministic tinyGLM gate produced identical tokens and neutral results
within normal run noise: medium C1 TTFT +0.4%, large C1 -0.3%, staggered large
C2 +0.1%. This is a correctness/dispatch gate, not a performance proxy: its
small routed-expert count does not reproduce the production model's launch
fan-out.

The promotion test used the complete 320B model on two GB10s, TP2, DFlash2
K=7, FP8 KV, 7,168 max batched tokens, mixed prefill scheduling, and the same
cooperative decode path in both arms. Both arms completed graph capture, the
20-request shape sweep, and a discarded four-resident 16K warm-up. The three
retained repetitions used byte-identical, cache-unique prompts; prefix-cache
hit rate remained zero. The harness's `--prompt-tokens 16384` fixture encoded
to 32,992-32,995 actual model tokens, so the retained table is honestly labeled
~33K rather than by its approximate input argument.

| Full-model ~33K actual-token metric | Existing path | Grouped path | Change |
|---|---:|---:|---:|
| C1 TTFT median | 25.739 s | 23.479 s | 8.8% faster |
| C1 effective prefill median | 1,281.9 tok/s | 1,405.3 tok/s | +9.6% |
| C2 staggered TTFT p50 median | 41.381 s | 38.850 s | 6.1% faster |
| C2 staggered TTFT p95 median | 50.034 s | 47.053 s | 6.0% faster |
| C2 staggered wall median | 50.536 s | 47.554 s | 5.9% faster |
| C2 aggregate effective prefill median | 1,305.8 tok/s | 1,387.7 tok/s | +6.3% |

All nine baseline requests and all nine candidate requests succeeded. The C1
one-token outputs matched across every repetition. C2 had the same two output
hashes across the experiment, with one baseline arm selecting the alternate
first token in one of three repetitions; the candidate was stable across all
three. Model-free kernel output remained bit-for-bit exact. This benchmark is
a performance/corruption gate, not a substitute for a broad quality suite.

The 65,536-route scratch is 1,207,980,548 bytes per rank. Reported KV capacity
changed from 1,268,292 to 1,219,512 tokens (3.8%). During candidate startup,
vLLM's graph memory accountant reported a nonsensical negative actual graph
pool after the persistent allocation; graph capture and serving still passed.
Treat that counter as an accounting anomaly until separately corrected.

Raw receipts retain the original harness-argument filename in
`benchmarks/receipts/{baseline,candidate}-16k-{c1,c2}-r*.json`; their
`requests[].prompt_tokens` fields are the authoritative actual counts.
The two-Spark launcher enables the path by default; set
`EXL3_GROUPED_PREFILL_K4=0` for immediate rollback.

## Provenance

See [PROVENANCE.md](PROVENANCE.md). The grouped planner and phase-wide
execution contract are original project work. The direct K4 trellis math and
M64 `cp.async` tile are derived from the exact upstream revisions recorded
there.
