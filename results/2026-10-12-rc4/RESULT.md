# 2026-10-12: faster single-stream decode, bit-exact with RC2 (engine sparkglm/atlas-20261012-rc4-layered)

The image is `846ee9e1e501`. It runs engine `5c523903` on `sparkglm/atlas-20261012-rc4-layered`: the upstream candidate
`upstream/glm53-flash-20261012-rc4` at `ad047638` plus the unchanged SparkGLM-only layer. Its tree (`ae5e79f8`) is
identical to `exp/prose-rc4` at `c6089633`, the build measured below.

## What changed since RC2

Ten engine changes, each behind its own switch, and one profile change, all on in the profiles. None of them changes a
target output bit.

- **All-reduce stages on the SMs** (`ATLAS_RDMA_ONESHOT_SM_STAGE=1`): the one-shot pair all-reduce stages its data with
  the SMs inside the PDL chain instead of a copy-engine hop.
- **MoE routing in two launches** (`ATLAS_GLM_MOE_ROUTE_FUSE=1`) and **the MLA layer front in two launches**
  (`ATLAS_GLM_MLA_FUSE=1`), down from a chain of small kernels each.
- **Verify dense FFN split across the pair** (`ATLAS_GLM_DENSE_TP_SPLIT=1`).
- **KDA record commits over a fixed grid** (`ATLAS_GLM_KDA_COMMIT_CTAS=96`).
- **Chunk-tiled MXFP8 twins** for q_b, o and the split verify head (`ATLAS_GLM_MXFP8_TILED=1`): each warp's chunk is
  one contiguous KiB. Same values into the same MMAs in the same order.
- **Drafter:** threshold top-k and parallel merges in the DFlash2 selector (`ATLAS_GLM_DRAFT_TP_TOPK=1`), uploads that
  do not drain the stream (`ATLAS_DFLASH_ASYNC_UPLOADS=1`), the selector's transition term weighted 0.6
  (`ATLAS_DFLASH2_EDGE=0.6`), NVFP4 twins of the drafter's delta projections (`ATLAS_DFLASH_CONV_TWIN=nvfp4`), and the
  confidence width's full-width probe every 256th step instead of every 64th (`ATLAS_DFLASH_CONF_PROBE=256`, a profile
  change). These move drafts and acceptance only; the canonical verify keeps the target's output exact.

## Measurements

Two DGX Sparks, production settings: `4x1m`, 0.91, display carveout, KV shard, disk prefix cache 48 GB. Greedy,
256 or 384 forced tokens, thinking off. The 12-prompt RC2 baseline is the RC2 engine with `ATLAS_DFLASH_CONF_PROBE=256`, the one
switch it already had.

| C1 decode | This release | RC2 |
|---|---:|---:|
| Prose, 12 prompts × 256 tokens, tok/s (`raw/multi-*.json`) | **45.57** | 42.82 |
| Code, 12 prompts × 256 tokens, tok/s | **64.91** | 61.38 |
| Strict prose bursts, 8 prompts × 384 tokens, tok/s (`raw/bursts-*.json`) | **47.9** | 44.8 |

Tokens per verify step are within 1% of RC2 (prose 2.50 vs 2.51, code 4.88 vs 4.85, bursts 2.51 vs 2.49): the gain is
step time, 52.6 ms against 55.9 ms per prose step in the bursts run. The bursts file has every token's arrival time.

**Bit-exact with RC2.**
- The prompt-logprob hash of a fixed 48,875-token text is `246cd1331b41`, unchanged (`raw/lp-rc4-on.json`).
- The 500 arithmetic problems and 24 two-hop 24K needles: 524 of 524 outputs are byte-identical to RC2's
  (`raw/qbig-rc4-on.jsonl` against `../2026-10-09-rc2/raw/qbig-rc2.jsonl`), so the scores are RC2's: 491/500 and 24/24.
- Determinism: four greedy prompts at C1, then all four concurrently, twice. Both C4 runs equal C1 on 4 of 4
  (`raw/rc4_on.json`).

**Deployed through LLooM** (production settings above):
- Four concurrent ~204K-token sessions, three turns each: 12/12 exact answers, turns 2+ served from the prefix cache
  (`raw/fill-prod.json`). The lowest MemAvailable on rank 0 during the fill was 4.4 GiB (`raw/prod-mem-rank0.log`).
- 45K-token multi-turn and a 35K-token replay: all correct, cached turns start in 0.7–0.9 s (`raw/pc-prod.json`).

**Release suite through LLooM** (the same runs as RC2's receipts, same settings; RC2's numbers from
`../2026-10-09-rc2/`):

| | This release | RC2 |
|---|---:|---:|
| RigMark decode, code / prose / structured, tok/s (median of 3, `raw/rig-decode.json`) | **72.3 / 38.3 / 91.7** | 65.5 / 36.5 / 88.8 |
| RigMark cold prefill 8K / 32K / 64K, TTFT (s) | 3.29 / 12.47 / 23.91 | 3.29 / 12.47 / 23.72 |
| RigMark staggered arrivals, prefill first: newcomer TTFT at 2 / 4 (s) (`raw/rig-conc.json`) | 2.80 / 3.54 | 2.80 / 3.54 |
| RigMark short code, 1 / 2 / 3 / 4 streams aggregate, tok/s (mean of 2 rounds) | 47.0 / **60.5** / 66.8 / **70.9** | 47.7 / 55.2 / 65.4 / 69.5 |
| Staggered C4 field guide, ~16K input (median of 5), s (`raw/field-16k/`) | **49.5** | 49.9 |
| Staggered C4 field guide, ~32K input (median of 3), s (`raw/field-32k/`) | **76.3** | 77.3 |
| Matrix (C1/C2 16K and 32K, C4 16K; sum of walls), s (`raw/matrix/`) | **132.1** | 136.3 |
| Quality probe: arithmetic / two-hop needle | 39/40 / 12/12 | 39/40 / 12/12 |

- RigMark samples, so its output text and lengths differ between runs and engines; its decode rows are not an
  equal-text comparison. Two structured runs came out compact (441 tokens: 91.3, 91.7 tok/s) and one pretty-printed
  (694 tokens: 95.5).
- Prefill does not change: the new kernels run in decode and verify. The 64K row is within run-to-run spread.
- The matrix's C4 16K case is 41.1 s against 40.5 s; the C1 and C2 cases are faster.

## Verify the receipts

The JSON files are the probes' own output, unedited.

```sh
cd results/2026-10-12-rc4 && sha256sum -c SHA256SUMS
```
