# Speculative slot-budget separation rejected — 2026-09-02

## Verdict

Reject `max_num_batched_tokens=7196` with
`max_num_scheduled_tokens=7168`. The extra worker allocation recovered the 28
input positions reserved by four DFlash2 K7 requests, but the changed maximum
shape cost more than those positions saved. Restore MNBT 7,168 with its default
scheduled ceiling.

## Byte-equivalent C4 gate

The control was the accepted 16 ms spinwait service. The candidate retained the
same image, vLLM build, mixed scheduler, DFlash2 K7/TP2, FP8 target KV, and four
33K requests arriving at 0/1/2/3 seconds. Both runs used the same prompt salt
across a restart and reported prompt-token counts
32,999 / 33,002 / 33,002 / 33,001. Every request completed 400 output tokens
without errors.

| Metric | 7,168 control | 7,196 buffer / 7,168 scheduled | Change |
| --- | ---: | ---: | ---: |
| Wall time | **154.041 s** | 158.265 s | **+2.74%** |
| Delivered throughput | **10.387 tok/s** | 10.110 tok/s | **-2.67%** |
| Aggregate decode | **13.990 tok/s** | 13.472 tok/s | **-3.70%** |
| TTFT lane 1 | 39.954 s | **39.792 s** | -0.40% |
| TTFT lane 2 | **65.624 s** | 69.483 s | +5.88% |
| TTFT lane 3 | **91.766 s** | 92.803 s | +1.13% |
| TTFT lane 4 | **112.775 s** | 113.417 s | +0.57% |

The candidate also changed the E2 allocation/profile maximum from 7,168 to
7,196 rows. The endpoint result shows that reclaiming at most 0.39% more input
work per full C4 slab does not amortize that shape change on GB10. Do not add a
separate scheduled-token knob for this profile.

Artifacts:

- `sparkglm-artifacts/spinwait16-candidate-23dd3b9676.json`
- `sparkglm-artifacts/spec-slot-budget-candidate-4b46eadc06.json`
