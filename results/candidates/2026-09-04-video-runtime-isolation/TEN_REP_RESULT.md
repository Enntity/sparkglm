# Ten repetitions per image: controlled video-workload comparison

## Answer

For this workload and these matched controls, the source-restored and preserved
images are effectively tied in headline speed. The earlier 5% throughput
deficit did **not** reproduce. This is evidence against a repeatable missing
engine optimization in this case, not proof that the binaries, dependencies,
all deployment conditions, or every latency metric are identical.

| Image — 10 measured repetitions each | Mean wall s | Median wall s | Wall range s | Mean delivered tok/s | Median delivered tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved video image | 88.835 | 88.422 | 85.674–91.355 | 23.372 | 23.356 |
| Source-restored image | 88.907 | 88.629 | 87.079–91.771 | 23.458 | 23.701 |

The rebuilt mean wall time is **0.08% longer**: only
0.072 seconds in an approximately 89-second workload.
Its mean delivered post-first-token rate is 0.37% higher.
These tiny differences are not optimization wins or established regressions.

## Controls and order

See the [declared protocol](TEN_REP_PROTOCOL.md) and
[reproducible summary](raw/ten-rep-summary.json). Twenty measured replays use
four approximately 15.8K-token prompts, arrivals at 0/1/2/3 seconds, and four
400-token outputs. The fixture deliberately preserves the posted-video prompt
bytes rather than replacing them with the newer calibrated G3 fixture.

Each image has two independent restarts, with five measured repetitions per
restart. Order is rebuilt / preserved / preserved / rebuilt. Every startup
passes shape and long-C4 warmup, then discards one exact replay. Warmups are
retained separately and excluded from the measured ten. No outliers were
removed and no failed runs were replaced.

Both ranks use the expected immutable image digest, CPU set `5-8,15-19`,
and `--num-gpu-blocks-override 650`, yielding 1,132,404 **reported** cache
tokens. The [startup excerpts](raw/fixed-cache-startup-excerpts.json) retain
loading, graph, cache and warmup events; each block also has a both-rank
identity/cache receipt. Reported cache capacity is not a promise that all
request shapes can use that many tokens.

| Chronological block | Mean wall s | Median wall s | Mean delivered tok/s |
| --- | ---: | ---: | ---: |
| Rebuilt A1 | 89.265 | 88.811 | 23.368 |
| Preserved B1 | 89.631 | 89.955 | 23.247 |
| Preserved B2 | 88.040 | 88.327 | 23.496 |
| Rebuilt A2 | 88.550 | 87.694 | 23.549 |

The first block pair has the rebuild 0.41% faster in mean wall time.
The reversed-order pair has it 0.58% slower.
The direction flips; there is no consistent five-percent penalty.

All twenty measured wall times, in seconds:

| Repetition within block | Rebuilt A1 | Preserved B1 | Preserved B2 | Rebuilt A2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 88.669 | 87.020 | 88.327 | 88.589 |
| 2 | 88.811 | 89.955 | 90.707 | 91.771 |
| 3 | 91.318 | 91.346 | 85.674 | 87.617 |
| 4 | 88.142 | 88.478 | 88.365 | 87.694 |
| 5 | 89.384 | 91.355 | 87.125 | 87.079 |

## Uncertainty and interpretation

The stratified within-block bootstrap gives a conditional 95% interval of
-1.38% to
+1.66% for the rebuilt mean wall-time change.
The corresponding throughput interval is
-1.52% to
+2.15%.
Both include zero.

These intervals condition on these four startups, assume independent
within-block resampling, and omit startup-level and serial-dependence
uncertainty. Two startups per image are not enough for a strong
startup-independent equivalence certification. ABBA balances a simple linear
time trend, not arbitrary external load changes. No equivalence margin or
new optimization acceptance is claimed.

## Secondary metrics and correctness boundaries

All **80 measured requests** completed, delivering **32,000 output tokens**.
Both images have zero observed prefix-cache hits and zero preemptions in
every measured replay. All actual prompt counts match the frozen fixture.

| Secondary metric — mean across ten runs | Preserved | Rebuilt |
| --- | ---: | ---: |
| First stream TTFT | 20.515 s | 20.836 s |
| Second stream TTFT | 30.017 s | 30.317 s |
| Third stream TTFT | 38.097 s | 38.163 s |
| Fourth stream TTFT | 44.176 s | 44.189 s |
| Per-replay visible SSE-gap p95 | 290.180 ms | 313.363 ms |
| Per-replay maximum visible SSE gap | 5.661 s | 5.495 s |
| Draft/request proposal rounds | 625.2 | 615.7 |

Do not call every latency statistic identical. The rebuild's first-stream TTFT
is about 0.321 seconds longer on average, and its mean per-replay SSE-gap p95
is about 23 ms longer. Those secondary observations are retained rather than
hidden behind the headline tie; this experiment does not establish a general
latency-regression claim. SSE gaps are **not** per-token latency: a speculative
event may contain several tokens. Proposal rounds sum per-request work, not
batched GPU forward passes.

Head-GPU median SM clocks are 2405–2418 MHz across the preserved runs and
2408–2414.5 MHz across rebuilt runs. Head temperatures span 58–84 C and
59–85 C, respectively. This does not identify a gross head-clock collapse;
it is not a complete two-rank thermal/throttling audit.

Every stream position has ten distinct output text hashes within each image,
despite greedy target sampling. The model does not produce bit-identical
continuations across these full-model repetitions, and speculative work varies.
The DFlash2 probabilistic-config label does not mean random draft sampling
for an all-greedy batch. These traces are throughput/completion evidence,
not a quality evaluation or proof of identical token choices.

## What this resolves—and does not

- The repaired build did not reproduce the earlier headline deficit with
  matched memory/CPU controls and stronger repeated sampling.
- We did find real runtime-dependency and automatic-memory-accounting
  differences; the [runtime investigation](RESULT.md) documents them.
  No missing optimized CUDA or RDMA library was established as the cause.
- Matching the pool and CPU placement is an intervention. These data do not
  identify which prior uncontrolled factor caused a particular slow sample,
  or prove that automatic sizing and unconstrained placement always perform
  identically. Earlier receipts remain intact.
- This is one posted-video C4 workload, not the full C1/C2/C4 16K/32K G3 matrix,
  a G4 quality/operation suite, or G5 release qualification. Unrelated host work
  and a small idle sidecar were preserved.
- No engine optimization, default, routing policy, or public artifact was
  promoted by this test. The fixed-cache override is diagnostic only.

## Reproduce the analysis and provenance

From the repository root:

```bash
python3 results/candidates/2026-09-04-video-runtime-isolation/probes/summarize-ten-reps.py \
  results/candidates/2026-09-04-video-runtime-isolation/raw
python3 scripts/qualification.py verify-all
```

The collector, campaign driver, filtering and statistical summarizer are
original diagnostic orchestration using existing Python/NVIDIA/PyTorch
interfaces. They introduce no kernel optimization or new upstream source.
Engine attribution and licenses remain governed by the existing provenance
ledger. Complete machine-local startup logs and binary images remain private;
the result bundle contains source, timing/text traces, hashes and sanitized
startup evidence, not weights or GPU binaries.
