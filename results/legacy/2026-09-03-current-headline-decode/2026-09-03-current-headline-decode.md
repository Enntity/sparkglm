# Current best-engine headline decode — 2026-09-03

## Verdict

The current appliance is at single-stream parity with Mia's current EXL3
reference and slightly ahead on prose, but it is not yet the aggregate decode
leader. On five-run medians it trails Mia's published simultaneous C2 and C4
aggregate results by approximately 5% and 6%, respectively.

The latency picture is materially better: this scheduler starts all
simultaneous streams promptly instead of leaving them queued. Its measured
median TTFT is 0.486 s at C2 and 0.772 s at C4. Those are 13.6x and 8.2x lower
than the 6.62 s and 6.30 s values in Mia's published sparkDash table, although
the TTFT comparison remains directional because the client harnesses differ.

## Frozen engine

- Repository commit: `495f27d9469ee3a4ba59617a3d808781326ac96f`
- Image tag: `sparkglm-vllm:exl3-m64-candidate-20260903a`
- Image ID, identical on both ranks:
  `sha256:20d3833c8e3e4b57d5dc74bc3f1c0fced820dfcbca605625790b3ff7e32dd4c4`
- Recipe stamp:
  `b4ed44e20214545f926f61efddbc52f2bbe58ca57ad2f645a8c7048bfeacfa47`
- Model: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- Hardware: two DGX Spark GB10 systems, TP2
- Draft: DFlash2 k=7, draft TP2
- EXL3 MoE: M64 fat expert tile, paired gate/up, fused activation
- Scheduler: 7,168 max batched tokens, 4 max sequences,
  `GLM53_MIXED_PREFILL_CHUNK=0`, phase interleave disabled
- KV: FP8 target cache, 10 GiB manual allocation, 319,326 token capacity
- Endpoint remained healthy after the suite.

Both ranks reported the same image ID and recipe stamp. Startup exercised all
42 EXL3 layers through the M64 fat path with no batched, sorted, or legacy
fallbacks.

## Protocol

The comparable synthetic headline gate matches Mia's published decode shape:

- warm, empty KV cache;
- count 1 through 200 prompt;
- temperature zero and thinking off;
- 400 forced output tokens;
- simultaneous C1, C2, and C4 requests; and
- five measured runs per concurrency.

`Stream tok/s` is the median per-request decode rate. `Aggregate tok/s` is the
sum of per-stream rates, matching Mia's published definition. `Delivery tok/s`
is the stricter client-observed throughput from the first request start until
the final stream completes; Mia does not publish this value.

## Results

| Concurrency | Our TTFT | Our stream tok/s | Our aggregate tok/s | Our delivery tok/s | Mia stream | Mia aggregate | Aggregate delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 0.464 s | **65.16** | **65.16** | n/a | 62.9 | 62.9 | **+3.6%** |
| C2 | 0.486 s | **49.12** | **98.23** | 94.54 | 51.7 | 103.3 | **-4.9%** |
| C4 | 0.772 s | **34.74** | **138.10** | 132.96 | 37.1 | 146.5 | **-5.7%** |

The aggregate samples, rather than only the medians, were:

- C2: 98.23, 98.42, 102.05, 81.24, and 79.10 tok/s;
- C4: 138.10, 137.85, 131.44, 146.18, and 144.44 tok/s.

Peak C4 reached 146.18 aggregate tok/s, effectively Mia's published 146.5,
but the median is the honest headline because it captures the observed
run-to-run variance.

## Single-stream prose and speculation

The five-run hash-map prose median was **29.29 tok/s**, with a 25.87-31.62
range, median TTFT 0.491 s, median DFlash2 acceptance 0.3792, and 2.655 accepted
tokens per target step. Mia's published same-protocol lab median is 27.1 tok/s,
so this result is **8.1% higher**.

The C1 structured median was 65.16 tok/s, acceptance 0.9776, and 6.843 accepted
tokens per target step. It is effectively equal to Mia's newer 65.1 tok/s lab
median (+0.1%).

The simultaneous-client harness does not currently snapshot DFlash counters,
so its zero-valued counter fields are unavailable data, not evidence of zero
speculation.

## Correctness and caveats

- All 40 headline requests returned HTTP 200, produced 400/400 tokens, finished
  by length, and contained no NaN. All 30 concurrent C2/C4 requests retained
  their own marker and contained no cross-request marker.
- The Paris, decimal-comparison, and sky-color semantic probes passed.
- All ten C2 outputs had the same hash. Nineteen of twenty C4 outputs matched;
  one response duplicated `37` in the count sequence. The first C1 sample did
  the same. The remaining structured samples did not. Temperature-zero output
  is not necessarily bit-identical across batch shapes on this checkpoint, so
  this is recorded rather than hidden; it did not produce corruption or a
  failed semantic gate.
- Mia's public table comes from sparkDash, whereas these simultaneous runs use
  the repository's streaming OpenAI harness. The token workload and metric
  definition match, making tok/s a useful comparison; cross-harness TTFT should
  be treated as directional until both engines are rerun through one client.
- This is deliberately a high-accept, short-prompt headline test. It does not
  replace the appliance's 16K-32K staggered C1/C2 acceptance gate.

## Interpretation

The engine is no longer losing at the single-stream kernel/model path. The
remaining headline gap is concurrency efficiency and variance: C2/C4 sometimes
reach reference speed, but do not hold it across runs. The next decode effort
should therefore instrument per-request DFlash acceptance under concurrency
and attribute the roughly 5-6% median loss between speculative verification,
TP2 collectives, and scheduler batch shape. Optimizing C1 again is unlikely to
close this particular gap.
