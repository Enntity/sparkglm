# Strict-skip parity receipt — 2026-09-02

This receipt compares the same physical 2× DGX Spark pair, EXL3 target and
DFlash2 drafter, TP=2, 307,200-token maximum context, 2,048-token scheduler
budget, four maximum sequences, fixed 10,200,547,328-byte FP8 KV cache, and
the same `real_workload_matrix.py` prompt salt
`2026-09-02-e3f690c`.

The published reference was `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks`
with vLLM source `487ecf187`. The candidate image was
`sparkglm-vllm:e3f690c`, image ID
`sha256:756e06f95e240719947aa5e90edd27c2cf643eb3857782e8292e23e06618ca6d`.
Both endpoints completed the same 20-request boot-shape warmup before their
recorded matrix.

## Root cause

The candidate's imported decode-floor v3 gate defaulted
`GLM53_MIXED_PREFILL_MAX_WAIT_MS` to 1,500 ms. With
`GLM53_MIXED_PREFILL_CHUNK=skip`, that deadline changed strict skip into
forced 512-token late-admission chunks. Repeated MoE expert-weight streaming
during those chunks collapsed the already-decoding request. The reference has
no deadline gate and keeps the waiting prefill deferred until the decode is
done.

The accepted correction defaults the deadline to `0` and passes it explicitly
to both containers. Positive deadlines remain an opt-in starvation/TTFT
tradeoff.

## C2 result

| Case | Engine | Prompt tokens | TTFT A / B (s) | Decode A / B (tok/s) | Worst gap A / B (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| medium | Candidate, bad 1,500 ms gate | 15,811 / 15,808 | 19.31 / 39.52 | 3.27 / 23.32 | 7.010 / 0.139 | 47.23 |
| medium | Published reference | 15,804 / 15,801 | 20.03 / 33.45 | 21.24 / 24.75 | 0.133 / 0.222 | 41.00 |
| medium | Candidate, strict skip | 15,811 / 15,808 | 18.04 / 29.26 | 24.50 / 30.06 | 0.131 / 0.138 | 36.35 |
| large | Candidate, bad 1,500 ms gate | 32,183 / 32,180 | 36.75 / 64.56 | 4.88 / 29.89 | 0.733 / 0.131 | 71.67 |
| large | Published reference | 32,176 / 32,173 | 42.61 / 72.13 | 30.35 / 29.25 | 0.136 / 0.129 | 79.28 |
| large | Candidate, strict skip | 32,183 / 32,180 | 35.38 / 61.91 | 26.09 / 28.45 | 0.131 / 0.130 | 69.12 |

Strict skip removes the qualitative failure: candidate request A improves
from 3.27 to 24.50 tok/s at medium and from 4.88 to 26.09 tok/s at large,
while worst gaps return to about 130 ms. Against the reference, corrected C2
wall time is 11.3% lower at medium and 12.8% lower at large; B TTFT is 12.5%
and 14.2% lower respectively.

## C1 result

| Case | Engine | Prompt tokens | TTFT (s) | Decode (tok/s) | Worst gap (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| medium | Candidate, bad gate | 15,811 | 17.09 | 22.40 | 0.143 | 19.90 |
| medium | Published reference | 15,804 | 20.38 | 26.20 | 0.131 | 22.78 |
| medium | Candidate, strict skip | 15,811 | 16.27 | 37.55 | 0.131 | 17.95 |
| large | Candidate, bad gate | 32,183 | 39.04 | 29.98 | 0.130 | 41.14 |
| large | Published reference | 32,176 | 37.06 | 32.28 | 0.130 | 39.01 |
| large | Candidate, strict skip | 32,183 | 31.87 | 25.43 | 0.130 | 34.35 |

Decode output hashes were not stable across engine restarts despite
temperature zero, so cross-run decode-rate percentages are not treated as
pure kernel comparisons. The no-collapse conclusion is supported by the C2
active-request gaps, TTFT, wall time, and completed-token counts rather than
by hash equality.

## Startup receipts

- First candidate cold launch: health in 980 s; graph capture 104 s; engine
  profile/KV/warmup 131 s.
- Candidate warm restart with durable caches: health in 500 s; graph capture
  38 s; engine profile/KV/warmup 52 s; shape warmup 20/20 in 31 s.
- Reference warm launch: target graph capture 51 s; engine profile/KV/warmup
  80 s; shape warmup 20/20 in 50 s.

The first candidate launch also had a seven-minute pre-loader JIT/cache miss.
It disappeared on the durable-cache restart; rank 0 then entered the shard
loader three seconds after DeepGEMM, matching the reference bootstrap shape.
