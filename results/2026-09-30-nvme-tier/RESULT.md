# 2026-09-30: prefix cache on disk (`PREFIX_CACHE_DIR`, `PREFIX_CACHE_GB`)

## Change

- **Installer only.** `PREFIX_CACHE_DIR` and `PREFIX_CACHE_GB` in `.env` switch
  on the engine's NVMe tier for evicted prefix-cache blocks and KDA snapshots
  (off unless the directory is set). Same engine as the
  [decode-step release](../2026-09-30-decode-step/RESULT.md).
- **Image:** `ghcr.io/enntity/atlas-sparkglm:d4a121b8371a`, built from a fresh
  clone of branch `release/nvme-tier`.

## Validation

Two DGX Sparks, 4 × 512K profile, `GPU_MEMORY_UTILIZATION=0.93`,
`PREFIX_CACHE_GB=48` on each Spark's internal NVMe, full pool. One pass.
Log in `raw/nvfull.log`, harness in `raw/`.

| Check | Result |
|---|---|
| Prompt log-probabilities of a 48,875-token prompt | hash `8c75d2886794`, equal to the decode-step release |
| Ten ~209K-token conversations in turn (about 2.1M tokens, more than the pool holds): turn 1, cold | 81.5–83.2 s each, 10/10 exact |
| Turn 2 of the three oldest, whose blocks had been evicted to disk | **1.21 / 1.44 / 1.47 s**, 3/3 exact |
| Turn 2 of the newest (still in GPU memory) | 1.04 s, exact |
| Restore of one conversation | about 13,100 blocks in 0.40 s (8.4 GB/s), 0 failures on either Spark |
| Image questions with known answers (colour halves, checkerboard) | 2/2 |
| Panic or ERROR lines, both Sparks | 0 |
| Lowest host memory available on rank 0 | 3.06 GB |

On an earlier capped-pool run of the same engine code (sparkglm-research,
`experiments/2026-09-30-backlog-and-decode-step`), every output after a restore
was byte-identical to a run where nothing was evicted.

## Limitations

- **One pass.** The capped-pool run had one slow turn (5.8 s against about
  1 s for the others) that has not been explained.
- **Cost.** At 48 GB the tier pins about 384 MB of host memory, which the KV
  pool gives up (about 64K tokens at 0.93).

## Verify the receipts

```sh
cd results/2026-09-30-nvme-tier && sha256sum -c SHA256SUMS
```
