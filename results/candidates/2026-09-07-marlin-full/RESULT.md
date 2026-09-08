# NVFP4 Marlin W4A16 full TP2 screen

Experimental measurement through an LLooM-managed runtime. Both target and draft use actual TP2. Exact configuration, image IDs and model revisions are recorded in qualification.json.

| Case | Retained repetitions | Median wall seconds | Median TTFT seconds |
| --- | ---: | ---: | ---: |
| c1-16k | 1 | 28.541 | 10.687 |
| c1-32k | 1 | 31.248 | 20.228 |
| c2-16k | 1 | 37.697 | 16.494 |
| c2-32k | 1 | 60.147 | 32.931 |
| c4-16k | 1 | 63.449 | 28.574 |

All requests completed 400 tokens with correct isolation. Corresponding repetition prompt hashes are checked in the final comparison bundle. Repetitions after the first use the already warmed process; the first full screen includes the discarded exact C4 warmup. The warmup script explicitly skips its optional sampler-cache postcondition.

Failed bounded semantic cases: none (16 of 16 passed). All raw outcomes remain part of the result. No broad quality, endurance, or default-promotion claim.

The manifest source revision identifies the benchmark checkout. Image source and immutable per-host images identify the actual engine. API speculative usage counters are unavailable; server acceptance logging, when present, is separate evidence.

Marlin consumes NVFP4 weights with BF16 activations (W4A16). Native NVFP4 uses W4A4. Different token hashes are not treated as a matched-arithmetic proof; the separate production-shaped numeric, graph and sanitizer evidence is in the Marlin G1/G2 bundle.
