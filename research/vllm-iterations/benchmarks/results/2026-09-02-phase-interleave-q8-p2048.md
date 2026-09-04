# Exclusive prefill/decode phase interleave — 2026-09-02

This is the acceptance receipt for the GLM-5.3-Flash EXL3 scheduler change on
the local 2x DGX Spark pair. The target workload is two medium or large prompts
arriving five seconds apart while the first request is decoding. Each request
must produce exactly 1,000 tokens and retain its own isolation marker without
emitting the peer's marker.

## Simple model

Strict skip protects an active decode by refusing every waiting prefill. That
keeps request A smooth, but request B cannot start its long prefill until A has
finished decoding. Mixed batching is not a viable fix on this GLM MoE path:
putting decode rows and a large prefill chunk in the same forward makes the
forward inherit the expensive prefill/expert-streaming shape.

The accepted scheduler instead alternates **exclusive** work types under
contention:

1. Run eight decode-only engine steps.
2. Run one prefill-only slab of at most 2,048 tokens.
3. Repeat until the waiting prompt is resident, then return to ordinary decode.

Both TP ranks execute the one scheduler output, so they make the same phase
choice. No model forward contains both prefill and decode rows.

## Controlled setup

- Target: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- Runtime: vLLM `0.1.dev18+ge3f690c8b`, TP=2 across two GB10s
- Candidate image: `sparkglm-vllm:phase-interleave-wip`
- Installed scheduler SHA-256 on both ranks:
  `fdcd726b494956c5d548d1456c48e4c0dc49f26051da18b1966128fba6327ca3`
- DFlash2: k=7, draft TP=1; target KV: FP8
- Limits: max model len 307,200; max sequences 4; scheduler budget 2,048;
  fixed KV allocation 10,200,547,328 bytes
- Harness: `benchmarks/sparkglm/staggered_openai.py`, C2, 5-second stagger,
  isolation prompts, 1,000 required output tokens, watchdog disabled
- Both modes used the same image and source. Only
  `GLM53_PHASE_INTERLEAVE` changed. Each mode was restarted, completed the
  same 20-request bounded shape warmup, and used the same per-size prompt salt.

## Accepted result: q8 / 2,048-token slab

| Prompt pair | Mode | B TTFT (s) | Aggregate delivery (tok/s) | Wall (s) | A max / p95 gap (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 15,803 + 15,800 | strict skip | 58.210 | 28.574 | 89.623 | 0.158 / 0.138 |
| 15,803 + 15,800 | phase q8/p2048 | 34.607 | 39.017 | 68.967 | 2.115 / 0.219 |
| 32,175 + 32,172 | strict skip | 85.945 | 23.013 | 122.301 | 0.146 / 0.136 |
| 32,175 + 32,172 | phase q8/p2048 | 74.260 | 27.736 | 107.438 | 2.069 / 2.037 |

At the medium shape, phase interleave lowers B TTFT by **40.5%**, raises
aggregate delivery by **36.6%**, and lowers wall time by **23.0%**. At the
large shape, it lowers B TTFT by **13.6%**, raises aggregate delivery by
**20.5%**, and lowers wall time by **12.2%**.

All eight control/candidate requests reached 1,000/1,000 output tokens with
`finish_reason=length`, retained their own marker, emitted no foreign marker,
and returned no error.

## Latency tradeoff

This is a throughput/fairness improvement, not free parallel execution. A
prefill slab temporarily pauses request A. At medium, A completion rises from
49.065 to 59.993 seconds; at large, from 62.582 to 86.128 seconds. The benefit
is that B starts substantially earlier and the pair finishes sooner. Set
`GLM53_PHASE_INTERLEAVE=0` to restore exact strict-skip behavior when a single
active stream's smoothness matters more than concurrent TTFT and total wall
time.

## Rejected smaller-slab tuning

The q2/512 schedule was tested to preserve the same nominal prefill/decode
work ratio with shorter pauses. Its first run exposed a one-time five-second
TileLang compile for the 512-row shape; the steady-state repeat used a fresh
prompt after compilation.

| Prompt pair | Mode | B TTFT (s) | Aggregate delivery (tok/s) | Wall (s) | A max / p95 gap (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 15,804 + 15,801 | phase q2/p512, warm | 39.281 | 35.474 | 74.197 | 1.013 / 0.825 |

The smaller slab halves the worst steady-state pause, but versus q8/p2048 it
raises wall time 7.6%, lowers aggregate delivery 9.1%, and raises B TTFT 13.5%.
It therefore fails the fastest-appliance acceptance goal and is not the
default.

## Reproduction

Run once with `GLM53_PHASE_INTERLEAVE=0`, restart and warm the service, then
run again with:

```text
GLM53_PHASE_INTERLEAVE=1
GLM53_PHASE_DECODE_STEPS=8
GLM53_PHASE_PREFILL_TOKENS=2048
```

For the medium case, invoke `staggered_openai.py` with:

```text
--concurrency 2 --stagger-ms 5000 --prompt-style isolation
--prompt-token-list 8000,8000
--output-token-list 1000,1000 --min-output-token-list 1000,1000
--disable-loop-watchdog --timeout-s 1200
```

Replace both prompt-list values with `16000` for the large case. Use a fresh
salt after every non-restarted tuning repeat so prefix caching cannot turn the
second measurement into a cache-hit benchmark.
