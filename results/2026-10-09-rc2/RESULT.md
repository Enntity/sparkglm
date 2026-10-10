# 2026-10-09: deterministic greedy output at any concurrency, faster MoE decode (engine sparkglm/atlas-20261009-rc2)

The image is `4046c81baa07`. It runs engine `f2b805e7`, which has the same tree as the layered integration branch
`sparkglm/atlas-20261009-rc2-layered`, built on upstream candidate `upstream/glm53-flash-20261009-rc2`.

## What changed since the 2026-10-05 release

- **KV shard (on by default).** Each Spark stores the latents of half the KV blocks. The pool is about 2.36M tokens,
  against about 1.38M before. Each request can use the model's full 1,048,576-token window: the default profile is
  now `4x1m`. The shard changes no output bit: the prompt-logprob hash is the same with the shard on and off.
- **Prefix cache on disk, beside the shard.** Each Spark spills and restores only the latents it owns.
- **Canonical verify** (`ATLAS_GLM_CANONICAL_VERIFY`, default on).
  - DFlash verifies a variable number of drafted tokens per step. The engine used to pick different kernels for each
    of those widths: projections, router, shared expert, dense FFN, and plain decode.
  - Width depends on the drafter and on what else is running, so the same request could get different greedy output
    at one stream than at four.
  - Every op now uses one row-invariant kernel family for every width from 1 to 32 rows.
- **Persistent MoE decode kernels** (`ATLAS_GLM_MOE_DECODE_PERSIST`, on in the profiles): one CTA per SM, prefetching
  the next expert tile. Bit-identical to the previous kernels.
- **KDA record commits:** one launch per sequence instead of one per layer. Bit-identical.
- **Copy drafts** (prompt lookup beside DFlash2). Streamed GLM text keeps the words tool/user/assistant. Tool
  arguments keep their schema types.

## Measurements

Two DGX Sparks, production settings: `4x1m`, 0.91, display carveout, KV shard, disk prefix cache 48 GB.

| | This release | 2026-10-05 release |
|---|---:|---:|
| RigMark decode, code / prose / structured, tok/s | 65.5 / 36.5 / 88.8 | 65.7 / 36.3 / 91.6 |
| RigMark cold prefill 8K / 32K / 64K, TTFT (s) | 3.29 / 12.47 / 23.72 | 3.28 / 12.35 / 23.71 |
| RigMark staggered arrivals, newcomer TTFT at 2 / 4 (s) | 2.80 / 3.54 | 2.86 / 3.51 |
| RigMark short code, 1 / 4 streams aggregate, tok/s | 47.7 / 69.5 | 45.8 / 67.0 (2026-09-30) |
| Staggered C4 field guide, ~16K input (median of 5), s | 49.9 | 49.3 |
| Staggered C4 field guide, ~32K input (median of 3), s | 77.3 | — |
| Matrix (C1/C2 16K and 32K, C4 16K; sum of walls), s | 136.3 | 136.7 |
| Four concurrent ~204K-token sessions, cached follow-ups | 12/12 and 8/8 | 12/12 |
| KV pool | 147,480 blocks (2.36M tokens) | 86,005 blocks per Spark (1.38M) |

**Structured decode.** RigMark's structured test asks for a JSON list. The model writes it either pretty-printed
(about 692 tokens) or compact (about 441 tokens).
- The 2026-10-05 median came from a pretty-printed run.
- All three runs on this release came out compact.
- Whitespace tokens are easy to draft, so the compact form scores lower tok/s, but it finishes in 5.7 s instead of
  8.2 s.
- On identical output text (the same output hash), this release runs at 88.8 tok/s against 87.2 for the
  2026-10-09 morning release.

**Single-stream decode over many prompts.** One prompt per workload is a poor measure across engines whose bits
differ, because the generated text changes and draft acceptance changes with it. So we ran 12 prose and 12 code
prompts, 256 tokens each, in two alternating rounds against the 2026-10-09 morning release (`raw/multi-*.json`):

| C1, 12 prompts per workload | This release | Morning release (3d795a21633d) |
|---|---:|---:|
| Prose tok/s | 42.95 / 42.89 | 41.16 / 41.18 |
| Code tok/s | 61.77 / 61.70 | 61.83 / 61.85 |

Per verify step this release is 2–3% faster: 59.9 vs 61.3 ms on prose and 82.7 vs 85.3 ms on code
(`raw/nsys-cmp.log`).

**Determinism.** We ran four greedy prompts at C1, then all four concurrently, twice, with batched verify, adaptive
draft width, copy drafts and the prefix cache on. Every C4 output equals the C1 output, and the two C4 runs agree
(`raw/rc2-spark-rc2-f2b805e7.json`). The previous release matched on 0 of 4 (`raw/rc2base-spark-rel-3d795.json`).

**Quality.**
- 500 multi-step arithmetic problems: 491/500.
- 24 two-hop 24K needles: 24/24 (`raw/qbig-rc2.jsonl`).
- Every engine configuration we ran scored between 484 and 495 on the arithmetic set: each arithmetic variant flips
  a different set of borderline items, so single runs can't rank engines.
- The log-likelihood of a fixed 48,875-token text moved by 0.000008 nats per token against the previous release.

**Against Mia's TensorFold v1.10 on the same pair** (`raw/tf110-*`):

| | This release | TensorFold v1.10 |
|---|---:|---:|
| Staggered C4 field guide at ~16K (s) | 49.9 | 60.2 |
| Staggered C4 field guide at ~32K (s) | 77.3 | 100.1 |
| Matrix (s) | 136.3 | 166.9 |

TensorFold is faster at single-stream decode on the same probe prompts: about 49.8 vs 42.2 tok/s on prose and 68.5 vs
54.6 on code. Its EXL3 experts are about 8% smaller per rank than our NVFP4 experts, and our MoE decode kernels already
read weights at the GB10 bandwidth roofline.

## Verify the receipts

The JSON files are RigMark's and the probes' own output, unedited. In the logs, home directories were replaced with
`/home/user`.

```sh
cd results/2026-10-09-rc2 && sha256sum -c SHA256SUMS
```
