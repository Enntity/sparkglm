# Decode step cut (engine sparkglm/atlas-20261003-stepcut)

Four exact engine options, on in both shipped profiles:

| Option | What it does |
|---|---|
| `ATLAS_GLM_MOE_DECODE_L2PF=1` | The routed-expert decode kernels also prefetch each weight stage's rows into L2 a few stages ahead. Prefetch hints only, so outputs are unchanged. |
| `ATLAS_GLM_MOE_STREAM_NOSYNC=1` | Batches of 9-32 rows (two to four streams verifying together) no longer copy the expert offsets back to the host, which drained the stream in every MoE layer. |
| `ATLAS_GLM_DRAFT_TP_BATCH=1` | The drafter's batched propose (two to four streams) is split across the two Sparks, as the single-stream propose already was. |
| `ATLAS_RDMA_ONESHOT=1`, `ATLAS_RDMA_PAIR_CHAIN=1` | One-shot RDMA writes for the pair's small all-reduces. |

Measured on the pair (2026-10-03, 4 × 512K profile, `ATLAS_DFLASH_FIRST_APPEND=none` in every arm): greedy text
identical to the options off on every prompt, and the prompt-logprob hash `8c75d2886794` unchanged.

| | Options off | Options on | Change |
|---|---:|---:|---:|
| 1 stream, prose (median of 5 × 384 tokens) | 40.26 tok/s | 41.02 tok/s | +1.9% |
| 2 streams, aggregate | 53.11 tok/s | 55.29 tok/s | +4.1% |
| 4 streams, aggregate | 65.97 tok/s | 69.40 tok/s | +5.2% |

Run-to-run noise: a repeated options-off run matched 1 and 2 streams within 0.1% and moved 4 streams by 1.8%.
A fifth option (`ATLAS_GLM_STEP_FUSE`, fused small kernels) was exact but 0.2-0.9% slower and is not shipped.
The 8 × 128K profile carries the same options; it was not measured separately. Raw probe output: `raw/`.
