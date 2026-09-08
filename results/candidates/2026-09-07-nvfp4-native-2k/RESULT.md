# Native NVFP4 with 2K prefill chunks full TP2 screen

Experimental measurement through an LLooM-managed runtime. Both target and draft use actual TP2. Exact configuration, image IDs and model revisions are recorded in qualification.json.

| Case | Retained repetitions | Median wall seconds | Median TTFT seconds |
| --- | ---: | ---: | ---: |
| c1-16k | 3 | 21.248 | 8.911 |
| c1-32k | 3 | 29.245 | 17.113 |
| c2-16k | 3 | 34.869 | 14.067 |
| c2-32k | 3 | 53.478 | 27.053 |
| c4-16k | 3 | 57.212 | 23.273 |

All requests completed 400 tokens with correct isolation. Corresponding repetition prompt hashes are checked in the final comparison bundle. Repetitions after the first use the already warmed process; the first full screen includes the discarded exact C4 warmup. The warmup script explicitly skips its optional sampler-cache postcondition.

Failed bounded semantic cases: arithmetic_91_89. All raw outcomes remain part of the result. No broad quality, endurance, or default-promotion claim.

The manifest source revision identifies the benchmark checkout. Image source and immutable per-host images identify the actual engine. API speculative usage counters are unavailable; server acceptance logging, when present, is separate evidence.


The final managed-runtime verification uses exact model identity at `/v1/models`: NVFP4 is healthy and stopped EXL3 is unhealthy on the same port. Both immutable images, shape warmup, and an attributed streamed tool call passed after a clean start. A prior live metadata update caused a restart that failed memory preflight; that failure is retained. Install profile updates while the appliance runtimes are stopped. These readiness checks were added after timing and do not change the GPU serving settings.
