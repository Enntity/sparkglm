# Latest-Mia apples-to-apples benchmark — 2026-09-03

## Question

Compare untouched Mia upstream at `eb0469f` with the SparkGLM branch derived
from that exact commit. The serving target is the real appliance workload:
four independent long-context requests arriving one second apart, rather than
an isolated short-prompt decode headline.

## Fixed workload and serving configuration

- 2x DGX Spark GB10, vLLM TP=2
- GLM-5.3-Flash EXL3 4 bpw target plus DFlash2 k=7 drafter
- FP8 target KV cache
- `max_num_seqs=4`, `max_num_batched_tokens=7168`
- four unique approximately 16K-token prompts, staggered by 1,000 ms
- 400 output tokens per request, ignore EOS, temperature 0, thinking disabled
- identical graph sizes: 1, 2, 4, 8, 16, 24, 32
- `GLM53_INDEXER_WORKSPACE=rightsize`, `GLM53_SPINWAIT_MS=16`
- prefix-cache collisions prevented with per-run salts; corresponding arms use
  the same salts

The reported aggregate output rate starts at the first emitted token and ends
when the final stream completes. Wall time starts when request 1 is submitted.
TTFT is measured independently from each request's scheduled submission time.

## Warm-up protocol

Every serving arm must complete graph capture and the 20-request DFlash2,
sampler, kpool, and concurrency shape sweep. It then receives a discarded
four-request 16K long-context run before retained measurements. Mia mixed and
SparkGLM also must prove all four requests resident during that long-context
gate. Strict `skip` is expected not to satisfy four-resident admission; its
gate completed with two running and two waiting, followed by an additional
discarded exact 4x16K/400-token run.

## Mia upstream controls

Mia `eb0469f` is the upstream `main` tip used for both controls. No source or
binary difference exists between the mixed and strict arms; only
`GLM53_MIXED_PREFILL_CHUNK` changes.

| arm | aggregate tok/s, runs | mean tok/s | wall, runs | mean wall | mean TTFT 1/2/3/4 |
|---|---:|---:|---:|---:|---:|
| Mia mixed (`0`) | 22.418, 21.170 | 21.794 | 95.212s, 99.222s | 97.217s | 23.924 / 35.027 / 46.261 / 51.280s |
| Mia strict (`skip`) | 15.143, 15.084 | 15.113 | 129.391s, 129.836s | 129.614s | 24.008 / 54.555 / 82.512 / 103.967s |

For this workload, mixed scheduling raises end-to-end aggregate output rate by
44.2%, cuts makespan by 25.0%, and cuts the fourth user's TTFT by 50.7%. Strict
mode keeps the currently active stream fast by making later prefills wait; it
is a decode-protection control, not the correct concurrent-appliance default.

Raw receipts:

- `receipts/mia-eb0469f-mixed-r1.json`
- `receipts/mia-eb0469f-mixed-r2.json`
- `receipts/mia-eb0469f-skip-r1.json`
- `receipts/mia-eb0469f-skip-r2.json`

Rendered captures are retained locally under `artifacts/`; they are excluded
from git because they are generated binary files.

## SparkGLM result

The candidate is commit `5940f05` on `sparkglm/gb10-optimized`, qualified as
image `sparkglm-mia:gb10-5940f05`. It uses the same Mia vLLM base revision as
the controls.

| arm | aggregate tok/s, runs | mean tok/s | wall, runs | mean wall | mean TTFT 1/2/3/4 |
|---|---:|---:|---:|---:|---:|
| Mia mixed (`0`) | 22.418, 21.170 | 21.794 | 95.212s, 99.222s | 97.217s | 23.924 / 35.027 / 46.261 / 51.280s |
| SparkGLM mixed (`0`) | 23.655, 23.158 | 23.407 | 92.212s, 91.113s | 91.663s | 23.466 / 33.506 / 45.015 / 49.110s |

Against its paired latest-Mia mixed control, SparkGLM raises aggregate output
rate by 7.4% and cuts wall time by 5.7%. Mean TTFT improves by 1.9%, 4.3%,
2.7%, and 4.2% for requests 1 through 4. Both paired runs win: +5.5% and
+9.4% aggregate output rate. Every stream produced all 400 requested tokens
without an error.

The retained traces are a two-run qualification, not a broad confidence
interval or a semantic quality evaluation. Outputs were coherent but not
byte-identical, which is expected when speculative and reduced-precision
execution paths differ. A separate quality suite is required before making a
quality-parity claim.

Full-engine schema-2 counters matched across both TP ranks after measurement:
1,344 prefill-layer calls, 1,191 sparse-kernel calls, and 78,577 paired expert
GEMM, direct, fused-activation, and scatter runs. No error, traceback, NaN, or
assertion was present in the serving logs.

Additional raw receipts:

- `receipts/sparkglm-5940f05-mixed-r1.json`
- `receipts/sparkglm-5940f05-mixed-r2.json`

## Qualification notes

- An attempted inline `skip` override was found to be shadowed by `.env`.
  Commit `2607089` preserves the caller-selected scheduler policy and adds a
  regression test. Live container environments were inspected for every arm.
- The first latest-Mia qualification image revealed that the bundled DeepGEMM
  rejects FP16 sparse-indexer logits. Commit `5940f05` therefore keeps the
  selector score dtype at FP32 by default; FP16 remains explicit opt-in until
  the native DeepGEMM path is compiled and proven end to end.
- Both nodes passed the EXL3 GPU self-check before `gb10-5940f05` launch. Live
  schema-2 diagnostics also proved paired expert GEMM and fused activation
  calls during full-engine execution.
