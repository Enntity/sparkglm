# 2026-09-29: prefix caching for multi-turn serving

## Why

The first Atlas release ([2026-09-29-atlas-merged](../2026-09-29-atlas-merged/RESULT.md))
had no prefix caching. An agent resends its whole conversation every turn,
so each turn of a 45K-token conversation re-prefilled it from scratch: about
20 seconds before the first token, every turn.

## What changed

- **Engine** ([`Enntity/atlas`](https://github.com/Enntity/atlas)
  `sparkglm/atlas-20260929b` @ `6a115315`). Two commits on the upstream
  candidate:
  - `0b3386d1` lets GLM-5's MLA layers take prefix-cache skips. Upstream
    Atlas declines the skip for every MLA model, because its generic MLA
    prefill only attends within the current chunk. GLM's prefill reads the
    full paged history, so for GLM a cached prefix is just a later chunk
    start.
  - `cc9467ed` adds `ATLAS_MARCONI_PREFILL_ONLY`, an opt-in that restores only
    block-aligned recurrent-state checkpoints written during prefill.
- **Profiles:** both profiles turn on `--enable-prefix-caching`, set
  `--ssm-cache-slots=16` and set `ATLAS_MARCONI_PREFILL_ONLY=1`.
- **Image:** `ghcr.io/enntity/atlas-sparkglm:2ba73a2d8aee`, built from a fresh
  clone of this repository (branch `fix/prefix-caching` @ `9bc77b2`).

## Results

Two DGX Sparks, 4 × 512K profile, greedy, thinking off unless noted. Receipts
are in `raw/`; the harness is in `raw/harness/`.

| Check | Result |
|---|---|
| Time to first token on turns 2–10 of a 30–48K-token conversation (median, 72 turns) | **0.81 s**, against 17.56 s with caching off |
| Immediate replay of a 35K-token prompt | 15.6 s cold, **0.81 s** replayed |
| Exact-answer retrieval over 40 turns in 4 documents, two runs | **40/40 and 40/40**; caching off 40/40 |
| Replay plus a 6-turn conversation | 8/8 |
| Four concurrent sessions, a tool-call turn, a partial hit after an edited middle section, thinking on | 23/23 |
| 400-token greedy answer, cold vs cache hit | byte-identical |
| Frozen matrix, cold (one pass) | 160.7 s: C1-16K 14.0, C1-32K 21.4, C2-16K 27.1, C2-32K 44.2, C4-16K 54.0 |
| Staggered C4 field guide, cold (the warm-up run) | 58.9 s |
| Lowest host memory available during the whole run (rank 0) | 7.85 GB |

## Correctness notes

- **Snapshot kinds.** Before `ATLAS_MARCONI_PREFILL_ONLY`, the engine could
  also restore its prompt-end, finish-leaf and decode-time snapshots. One
  restore of a prompt-end snapshot answered a turn with the neighbouring
  record's code, and did so identically on two server starts
  (`raw/pc-before-prefill-only.json`, turn 5). A cold prefill of the same
  prompt answered correctly.
- **An earlier non-deterministic miss.** One 40-question run in that earlier
  configuration missed a single question. Its receipt was overwritten by a 40/40
  re-run (`raw/acc-before-prefill-only.json`). With prefill-only restores, every
  check above passed.
- **Cost of prefill-only restores.** A warm turn replays about 50 tokens, from
  the last block boundary before the match to the end of the prompt. Each
  request uses one of the 16 snapshot slots instead of three.

## Limitations

- **Smaller KV pool.** Each 74 MB snapshot is held in GPU memory, and with
  prefix sharing on, every block also stores the sparse indexer's raw tails.
  The pool on this start was 341,888 tokens, against 866,208 without
  caching. The engine sizes the pool from memory in use at startup, and this
  run shared rank 0 with other services.
- **Slower concurrent prefill.** With caching on, the engine disables its
  fused prefill/decode path. Cold C4-16K took 54.0 s, against 45–51 s in the
  previous release; the cold staggered C4 is about the same.
- **Warm repeats.** The three field-guide repeats (32.6 s each) reuse the
  warm-up's prompts, so they measure cache hits, not a cold workload.
- **Scope.** One matrix pass; memory sampled on rank 0 only.
- **Scrubbed addresses.** Private addresses in `validate.log` and
  `validate.sh` were replaced with documentation addresses (192.0.2.x) and
  `/home/user`.

## Verify the receipts

```sh
cd results/2026-09-29-prefix-caching && sha256sum -c SHA256SUMS
```
