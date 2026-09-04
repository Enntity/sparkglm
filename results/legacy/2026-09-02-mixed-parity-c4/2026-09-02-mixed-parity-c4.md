# Mixed-scheduler parity profile — 2026-09-02

This receipt promotes the work-conserving mixed scheduler profile for the
opinionated 2x DGX Spark GLM-5.3-Flash appliance. It compares four requests
arriving at 0/1/2/3 seconds. Every request completed exactly 400 output tokens.

## Change

The candidate retained the accepted FP16 sparse-indexer selector image and
changed only the serving envelope from the earlier exclusive-phase profile:

| Setting | Earlier phase profile | Mixed parity profile |
| --- | --- | --- |
| `GLM53_MIXED_PREFILL_CHUNK` | `skip` | `0` |
| `GLM53_PHASE_INTERLEAVE` | `1` | `0` |
| `MAX_NUM_BATCHED_TOKENS` | 2,048 | 7,168 |
| `DFLASH_DRAFT_TP` | 1 | 2 |
| `GLM53_INDEXER_WORKSPACE` | `stock` | `rightsize` |
| `GPU_MEM_UTIL` | 0.79 | 0.87 |
| `MAX_MODEL_LEN` | 307,200 | 1,000,000 |

Both candidate ranks used image digest
`sha256:107d899515629f168fdd5f7bac3e967bc3d0fe768124bafd9322c7a3a8c4adc4`
and vLLM `0.1.dev47+g6a49e49e7`. Startup reported the E2 direct kernel for all
42 fat-expert layers, 19.51 GiB KV cache per rank, and 1,343,205 tokens of KV
capacity. The right-sized indexer workspace reclaimed about 4.9 GiB per rank
at the 1M context limit.

## Results

`Delivery tok/s` is total completed tokens divided by wall time from the first
scheduled request through the final completion. It is the primary appliance
metric here. `Aggregate decode tok/s` starts its denominator at the earliest
first token, so an earlier first response can lower that number even when total
wall time improves.

| Shape and engine | Wall (s) | Delivery tok/s | Aggregate decode tok/s | TTFT by arrival (s) |
| --- | ---: | ---: | ---: | --- |
| 54-57 prompt tokens, Mia mixed | 44.678 | 35.812 | 36.440 | 0.879 / 0.564 / 0.607 / 0.824 |
| 54-57, earlier phase profile | 43.630 | 36.672 | 36.845 | 0.313 / 1.634 / 0.636 / 2.254 |
| 54-57, mixed parity candidate | **40.298** | **39.704** | **40.020** | 0.417 / 0.502 / 0.742 / 0.755 |
| ~32,990, Mia mixed | 162.414 | 9.851 | 13.610 | 45.142 / 75.900 / 98.502 / 120.207 |
| ~32,990, earlier phase profile | 199.947 | 8.002 | 9.704 | 35.470 / 80.971 / 129.263 / 174.140 |
| ~33,002, mixed parity warm | 164.726 | 9.713 | 13.542 | 46.872 / 72.876 / 99.616 / 121.117 |
| ~33,002, mixed parity recorded | **159.581** | **10.026** | 12.939 | **36.233 / 64.376 / 92.662 / 115.661** |

The long candidate warm and recorded runs used different salts at the start of
the synthetic context. Prefix caching therefore could not turn the repeat into
a cache-hit benchmark. All prompt counts above are the server-reported token
counts, not byte or word estimates.

## Decision

The mixed parity profile passes this C4 gate. Against the earlier phase profile,
the recorded 33K run lowers wall time 20.2% and raises delivered-token throughput
25.3%. Against Mia mixed, it lowers wall time 1.7% and raises delivery 1.8%; this
single comparison establishes parity, not a durable kernel lead.

The earlier q8/p2048 phase policy remains an opt-in C2/decode-protection mode.
Its global 2,048-token budget let the oldest prefill consume each exclusive slab,
which serialized the later C4 prefills. A future scheduler improvement should be
benchmarked against this mixed profile and must beat its wall time and TTFT while
recovering incumbent decode smoothness.
