# Native NVFP4 with MXFP8 draft tuned TP2 screen

Experimental complete configuration; E3 is disabled. TP2 applies to both target and draft.

| Case | Repetitions | Median wall seconds | Median TTFT seconds |
| --- | ---: | ---: | ---: |
| c1-16k | 3 | 23.279 | 9.847 |
| c1-32k | 3 | 31.015 | 18.651 |
| c2-16k | 3 | 38.788 | 15.505 |
| c2-32k | 3 | 53.396 | 30.665 |
| c4-16k | 3 | 63.791 | 26.415 |

Configuration: 64K context, 7 speculative tokens, MXFP8 DFlash2 draft, 1024-token prefill chunks and 9 GiB KV per rank. Reported logical cache: 192275 tokens.

API accepted/rejected prediction counters are zero because this endpoint does not expose those usage fields; they are not measurements of actual draft acceptance. Where retained, server SpecDecoding metrics provide actual acceptance evidence.

The benchmark manifest source revision identifies the harness checkout; immutable image IDs and image source revision are separately recorded in qualification.json. Warmup sampler-cache postcondition was explicitly skipped.

No default promotion or general model-quality claim. The 91x89 arithmetic check also fails on the reference; this does not excuse or erase the raw failure.
