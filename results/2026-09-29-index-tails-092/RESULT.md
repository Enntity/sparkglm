# 2026-09-29: index tails in shared slots, 0.92 memory utilization

## Change

- **Engine:** [`Enntity/atlas`](https://github.com/Enntity/atlas)
  `sparkglm/atlas-20260929c` @ `0965f8d5`. On top of the
  [prefix-caching release](../2026-09-29-prefix-caching/RESULT.md), `2f9d940b`
  keeps the sparse indexer's raw tails in a small pool of shared slots, not
  one per KV block. Restores are block-aligned, and each 4-token pool sits
  inside one 16-token block, so a cached block never needs its tail again.
  - KV cost per token and rank drops from 12,228 B to 6,596 B.
  - A departing sequence returns its lent tails when it frees its blocks.
- **Knob:** `SPARKGLM_GPU_MEMORY_UTILIZATION` (`GPU_MEMORY_UTILIZATION` in
  `.env`, validated 0.80–0.95). This run used 0.92. The shipped default
  stays 0.88.
- **Image:** `ghcr.io/enntity/atlas-sparkglm:42fd6b6147be`, built from a fresh
  clone of branch `feat/tails-and-memory`.

## Results

Two DGX Sparks, 4 × 512K profile, `GPU_MEMORY_UTILIZATION=0.92`. Receipts are
in `raw/` and the harness in `raw/harness/`. Rank 0 still hosted the LLooM
gateway, with other services, for part of the run.

| Check | Result |
|---|---|
| KV pool (shared by all requests and the prefix cache) | **1,386,768 tokens**; 364,480 in the prefix-caching release at 0.88 |
| Four concurrent ~204K-token sessions, 3 turns each (~817K tokens live) | 12/12 exact answers; cold prefill 204K tokens in ~110 s (~1,860 tok/s) per request, queued one after another |
| Multi-turn, 30–48K tokens: time to first token on turns 2+ (median) | 0.81 s |
| Exact-answer checks (replay, 6-turn, concurrent, tool, partial hit, thinking, 40 retrieval turns) | 71/71 |
| 400-token greedy answer, cold vs cache hit | byte-identical |
| Frozen matrix, cold (one pass) | 160.3 s |
| Staggered C4 field guide, cold (the warm-up run) | 58.1 s |
| Lowest host memory available | 3.1 GB on rank 0, 6.2 GB on rank 1 |

In the fill test, warm turns that queued behind another session's cold 204K
prefill took up to about 110 s to start. Unqueued warm turns started in
1.5–13 s.

## Limitations

- **Headroom.** At 0.92, rank 0 got down to 3.1 GB free while sharing the host
  with other services. The shipped default stays 0.88; raise it only on hosts
  that run nothing else.
- **Scope.** One matrix pass and one fill run.
- **Scrubbed addresses.** Private addresses in `validate4.log` and
  `validate4.sh` were replaced with documentation addresses (192.0.2.x) and
  `/home/user`.

## Verify the receipts

```sh
cd results/2026-09-29-index-tails-092 && sha256sum -c SHA256SUMS
```
