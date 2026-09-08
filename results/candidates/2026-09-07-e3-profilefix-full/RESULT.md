# Corrected E3 concurrent EXL3 full TP2 campaign

Experimental measurement through an LLooM-managed runtime. Both target and draft use actual TP2. Exact configuration, image IDs and model revisions are recorded in qualification.json.

| Case | Retained repetitions | Median wall seconds | Median TTFT seconds |
| --- | ---: | ---: | ---: |
| c1-16k | 3 | 23.371 | 10.949 |
| c1-32k | 3 | 34.624 | 24.884 |
| c2-16k | 3 | 36.846 | 19.072 |
| c2-32k | 3 | 55.906 | 32.161 |
| c4-16k | 3 | 67.312 | 29.638 |

All requests completed 400 tokens with correct isolation. Corresponding repetition prompt hashes are checked in the final comparison bundle. Repetitions after the first use the already warmed process; the first full screen includes the discarded exact C4 warmup. The warmup script explicitly skips its optional sampler-cache postcondition.

Failed bounded semantic cases: arithmetic_123_47, arithmetic_91_89. These failures remain part of the result. No broad quality, endurance, or default-promotion claim.

The manifest source revision identifies the benchmark checkout. Image source and immutable per-host images identify the actual engine. API speculative usage counters are unavailable; server acceptance logging, when present, is separate evidence.

This candidate reserves E3 scratch and runs reference profiling before automatic KV sizing. Its corrected profiling adapter changes allocation accounting, not the copied numerical kernel. Concurrent dispatch preserves the reference path for solo prefill; 32 temporary expert rows and 7168-token chunks are explicit settings.
