# Results map

This page summarizes primary evidence. The machine-verifiable canonical index
is `results/index.json`; methodology and qualification rules are in
`docs/METHODOLOGY.md` and `docs/QUALIFICATION.md`.

## Current runnable line

### Latest-Mia apples-to-apples comparison

The retained comparison used four independent approximately 16K-token prompts,
one-second staggered arrivals, 400 requested output tokens, identical graph and
cache settings, and a discarded full-workload warmup before measurement.

| Comparison | Aggregate output | Wall | TTFT |
| --- | ---: | ---: | ---: |
| Mia mixed versus Mia strict skip | +44.2% | -25.0% | fourth request -50.7% |
| SparkGLM `5940f05` versus Mia mixed | +7.4% | -5.7% | requests 1-4 improved 1.9-4.3% |

Primary bundle: `results/legacy/2026-09-03-latest-mia-apples-to-apples/`.

### GPU-resident grouped prefill

The complete 320B checkpoint test used approximately 33K actual tokens per
prompt, TP2, DFlash2 k=7, FP8 target KV, mixed scheduling, identical cooperative
decode on both arms, and three retained paired repetitions.

| Metric | Change |
| --- | ---: |
| C1 TTFT | -8.8% |
| C1 effective prefill | +9.6% |
| staggered C2 TTFT p50 | -6.1% |
| staggered C2 wall | -5.9% |
| staggered C2 aggregate effective prefill | +6.3% |

Primary bundle: `results/legacy/2026-09-03-grouped-prefill-k4/`.

### Cooperative decode

The exact-shape kernel A/B improved 5.5-13.9% for one to sixteen tokens. The
warmed tinyGLM endpoint gate improved C1 by 5.5% and staggered C4 by 2.9%; long
C2 was -0.2%. Those results prove the mechanism and exact token agreement in
the gate, but do not establish a universal full-checkpoint serving gain.

Primary bundle: `results/legacy/2026-09-03-cooperative-decode/`.

## Inherited major result

Mia/plotarmordev's E2 fat-expert path improved fully uncached long-context
prefill by approximately 20-21%. This is an upstream result retained in the
recipe and must not be presented as original SparkGLM work. See
`docs/upstream/MIA_RECIPE_README.md`.

## Historical and negative results

The older campaign is indexed under `results/legacy/`.
It records rejected TP row sharding, fused score/top-k, speculative slot
padding, grouped E2 dispatch, M32 and alternate M64 tiles, FlashKDA integration,
hot-expert caches, and other candidates. Patch mailboxes retain the code path
for inspection.

The final uncommitted follow-up experiments are retained under
`research/vllm-iterations/current-followups/` and explicitly remain rejected or
research-only.

The Atlas-native measurements and status reports are under
`research/atlas/docs/`. They are not comparable to the current vLLM line
without following each report's exact scope and precision.

## Claiming rules

- Use only a number whose workload, precision, revision, warmup, and metric are
  stated together.
- Do not add independent percentage improvements arithmetically.
- Do not use tinyGLM throughput as a prediction of full-model throughput.
- Do not treat a kernel microbenchmark as an endpoint result.
- Do not present a rejected experiment's best sample as a retained result.
- Distinguish scheduler queueing improvements from raw GPU compute speedups.
