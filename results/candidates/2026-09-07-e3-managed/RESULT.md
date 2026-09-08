# Broad E3 dispatch rejected after the full-model screen

The >=4096-token policy passed isolated tinyGLM and operator checks, but failed the full-model performance screen. It remains disabled by default. The primary 32K C2 wall-time result is noisy, while protected 16K C1 work regresses. C4 is promising enough to motivate a separate mixed-workload dispatch candidate.

## Three paired repetitions

Each pair used identical exact prompt tokens, prompt hashes, arrival offsets and 400 output tokens per request. Both runtimes were owned by LLooM; each cold start was followed by shape warmup and a discarded C4 workload. Ratios below are reference wall time divided by candidate wall time, so values above 1 favor E3.

| Case | Reference median wall (s) | E3 median wall (s) | Median paired speed ratio | Paired range |
| --- | ---: | ---: | ---: | --- |
| c1-16k | 24.51 | 25.34 | 0.907 | 0.803–1.002 |
| c1-32k | 35.61 | 34.21 | 1.036 | 1.012–1.064 |
| c2-16k | 40.02 | 39.95 | 1.002 | 0.965–1.045 |
| c2-32k | 60.90 | 59.33 | 1.045 | 0.933–1.087 |
| c4-16k | 78.54 | 62.63 | 1.260 | 1.090–1.293 |

The sub-5% primary median is not a qualified improvement: three pairs do not establish a nonzero confidence interval. All requests completed with their own marker and no foreign markers. Output hashes differ between arms. Per-request TTFT, SSE gaps, completion times and hashes are retained in the raw receipts; SSE gaps are not per-token latency.

## Semantics and limits

The bounded suite passed tools, parallel tool results, structured output, reasoning, long prefix retrieval/isolation, stop, image color recognition and post-cancellation recovery. Both the reference and final E3 run answered 91×89 incorrectly. The earlier E3 run also missed 123×47; that error did not repeat after a fresh candidate start. These short prompts do not activate E3. This is a retained quality/nondeterminism limitation, not a general quality certification or proof that every difference is caused by E3.

The runs used the same 0.87 total GPU-memory allocation policy and 1M model limit, but automatic KV allocation differed: about 1.145M reference tokens versus 1.232M E3 tokens. This is an appliance-policy screen, not an explicitly equal-KV-byte operator comparison. The initial candidate cold start failed while a large checkpoint downloader occupied shared memory; stopping both ranks through LLooM and retrying after the transfer completed restored startup. A small later drafter acquisition and verification finished before the final retained matrix. No thermal slowdown was observed in the sampled telemetry.

The client manifest source_revision identifies the benchmark checkout, not the serving image source. The serving E3 source is 80b85461737059fd00ba8c7eedafd9749e267628; its two image IDs and serving hashes are in qualification.json. The preserved reference image is pinned separately. The authenticated streamed-tool canary and gateway attribution verify local glm53-flash-exl3 routing. G5 release/endurance was not attempted.

## Earlier integration evidence

The isolated three-repetition tinyGLM gate preserved all token signatures, and CUDA memcheck reported zero errors for all six top-8-cardinality cases. Universal and 2048-token dispatch were rejected separately. An earlier tinyGLM warmup during a large checkpoint transfer had one mixed-C4 signature difference; it remains retained and background-load batch invariance is unproven.
