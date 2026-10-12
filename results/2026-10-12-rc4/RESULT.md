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

Rows of the README not listed here were not re-run for this release.

## Verify the receipts

The JSON files are the probes' own output, unedited.

```sh
cd results/2026-10-12-rc4 && sha256sum -c SHA256SUMS
```
