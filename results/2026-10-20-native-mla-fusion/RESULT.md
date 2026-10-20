# SparkGLM TP2: native FP8 MLA fusion

2026-10-20. Restored the existing MLA-front fusion for native FP8 block weights on TP2. The existing switch stays on in the RC capacity profile. No weight conversion, FP4 rounding policy, drafter policy, or serving limit changed. Strict native-MTP work is retained.

| Workload | Previous Spark tok/s | Fused Spark tok/s | Gain | Saved NVFP4 tok/s | Versus NVFP4 |
|---|---:|---:|---:|---:|---:|
| Prose | 37.486 | 37.656 | +0.45% | 39.344 | -4.29% |
| Code | 66.595 | 67.147 | +0.83% | 72.252 | -7.07% |
| Structured | 89.149 | 91.107 | +2.20% | 91.944 | -0.91% |

Medians of three repetitions, both ranks uncapped. Every measured repetition:

| Workload | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Prose | 38.584 | 37.656 | 37.224 |
| Code | 69.234 | 66.786 | 67.147 |
| Structured | 89.102 | 91.238 | 91.107 |

## Exactness and focused validation

- GPU tests passed with PDL enabled: native raw projections before normalization, with and without prefetch; the native full front including normalized latents, index data and FP8 KV-cache bytes; and the original MXFP8 front. All widths 1–32, four start positions, unsharded and both TP2 ownership cases. The full-front checks cover 384 cases per format. Native scales include non-power-of-two FP32 values.
- Five served fusion-off/on checks passed: arithmetic, JSON, prose, code and structured output; all 373 token IDs, response text, reasoning text and finish reasons matched exactly.
- All nine timed outputs matched the previous Spark receipt in text hash, reasoning hash, token counts and finish reason. Per-request details are in raw/comparison-summary.json.
- Two targeted Rust CPU tests and 27 product installation tests passed. Three GPU tests passed. The first GPU attempt failed before MLA because the builder exposed link-time CUDA stubs; rerunning with the actual driver libraries passed. Both attempts are retained.
- All nine timed completions stopped naturally and passed the existing completion gates, including JSON checks. No broader quality, context, concurrency, or prefill campaign was run.

## What changed

The fused front now accepts the checkpoint’s native FP8 q_a/kv_a twins and reuses the exact existing FP8-block CTA arithmetic. It preserves the scale application, accumulation order, BF16 rounding, and fused-mid cache/norm code. Native scale-prefetch lengths use FP32 128x128 block sizes. Native activation is limited to TP2 and excludes packed prefill; unsupported layouts keep the existing path.

## Measurement and provenance

The same RigMark prompts, nonce strings, temperature 0, top_p 1, low reasoning, seed 20260905, 4,096-token allowance and streaming protocol were used. Decode rate is (completion_tokens - 1)/(last output time - first output time), excluding prefill. Effective ATLAS environment settings exactly match the previous Spark timing run. Each run uses FP4-Local, Spark mode, DFlash2, 4x1m, memory utilization 0.91, KV sharding, display carveout and 48 GiB disk prefix caching.

The NVFP4 column reuses the best saved three-run median per workload: October 12 RC4 for code, October 15 for prose and structured. The older checkpoint/build generated different outputs; code used a 4x512k capacity profile. These are historical same-pair comparisons, not a new old-build A/B or a universal performance guarantee.

Nine-request wrapper including its warmup: 191.7 seconds.

Engine commit: `1b186c435a8fd2e2b9da7c41c8c0375a7d90834f`.
Image: `sha256:1e87dbed59f99ef9f26d9fd831354314e91664bfc47136df79ac191a31c6b923`.
Image tag: `localhost/atlas-sparkglm:native-mla-fuse-a7ac2c046d78`.
Binary SHA256: `7c990fc44d5f2f3764d99a72faaf1b51d26490e6209e3f09ae17efaf02a7aef5`.

## Pair state

The validated RC is installed as managed Spark serving on both 01/02 nodes. The normal clock caps are restored; the reservation is released and the gateway canary returned 42. This is a local unpublished RC.

Raw requests, completions, comparison data, GPU tests, build identity, launch settings, and restoration evidence are under raw/. SHA256SUMS covers every artifact.
