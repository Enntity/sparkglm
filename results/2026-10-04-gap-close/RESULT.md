# Gap close (engine sparkglm/atlas-20261004-gapclose)

Two more exact engine options, on in both shipped profiles:

| Option | What it does |
|---|---|
| `ATLAS_GLM_HC_SEAM_ILP=3` | Load-batched twins of the mHC decode seam kernels (partial rows and finalize) on the BF16 mix weight. Same arithmetic and reduction order; a GB10 bench matches the original kernels bit for bit at 1-32 rows and runs 44.7 -> 39.4 us per seam at 1-8 rows. |
| `ATLAS_GLM_DRAFT_TP_CTX=1`, `ATLAS_DFLASH_CTX_ASYNC_POS=1` | The drafter's context append is split across the two Sparks like its MLP and head, and its positions upload in stream order. |

Measured on the pair (2026-10-04, 4 × 512K profile with the step-cut options, `ATLAS_DFLASH_FIRST_APPEND=none`):
greedy text identical with the options off, prompt-logprob hash `8c75d2886794` unchanged.

| | Options off | Options on | Change |
|---|---:|---:|---:|
| 1 stream, prose (median of 5 × 384 tokens) | 40.83 tok/s | 41.75 tok/s | +2.3% |
| 2 streams, aggregate | 55.18 tok/s | 56.24 tok/s | +1.9% |
| 4 streams, aggregate | 69.27 tok/s | 69.73 tok/s | +0.7% |

A repeated options-off run moved by 0.2-0.5%. Two other candidates were exact but slower and are not shipped: a
side-stream fork of independent layer chains (`ATLAS_GLM_LAYER_FORK`, about 20% slower: the fork and join break the
main stream's programmatic dependent launches) and an L2 weight prefetch ahead of use (`ATLAS_GLM_L2_AHEAD`, neutral
to slightly slower). Raw probe output and the GPU benches are in `raw/`.
