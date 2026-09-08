# EXL3 with MXFP8 draft tuned TP2 screen

Experimental complete configuration; E3 is disabled. TP2 applies to both target and draft.

| Case | Repetitions | Median wall seconds | Median TTFT seconds |
| --- | ---: | ---: | ---: |
| c1-16k | 1 | 24.155 | 12.021 |
| c1-32k | 1 | 35.387 | 23.482 |
| c2-16k | 1 | 40.626 | 21.706 |
| c2-32k | 1 | 61.624 | 38.327 |
| c4-16k | 1 | 74.022 | 35.849 |

Configuration: 64K context, 7 speculative tokens, MXFP8 DFlash2 draft, 7168-token prefill chunks and automatic KV allocation. Expert temporary rows: 128.

API accepted/rejected prediction counters are zero because this endpoint does not expose those usage fields; they are not measurements of actual draft acceptance. Where retained, server SpecDecoding metrics provide actual acceptance evidence.

The benchmark manifest source revision identifies the harness checkout; immutable image IDs and image source revision are separately recorded in qualification.json. Warmup sampler-cache postcondition was explicitly skipped.

No default promotion or general model-quality claim. The 91x89 arithmetic check also fails on the reference; this does not excuse or erase the raw failure.

Failed bounded cases: arithmetic_123_47, arithmetic_91_89.
