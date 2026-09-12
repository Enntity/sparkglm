# Active Atlas research: day-two continuation

Atlas remains an **experimental AGPL engine**, separate from SparkGLM's
recommended vLLM path. The complete chronology, failed screens, bounded
qualification and raw selected receipts are in the
[day-two result bundle](../../results/candidates/2026-09-11-atlas-day2/RESULT.md),
bound by its [G0-only qualification](../../results/candidates/2026-09-11-atlas-day2/qualification.json).

| Workload | Early day-two Atlas | Latest measured Atlas | Retained vLLM |
| --- | ---: | ---: | ---: |
| First field prompt client TTFT, 15,807 input / 8 output | r14: 12.775480s | r23: 11.259161s | 10.848938s |
| Four field prompts, 1600 outputs, total wall | r17: 205.467964s | r23: 137.129172s | 90.500192s |
| Outputs / total wall second | 7.787102 | 11.667831 | 17.679521 |

These are retained observations, not an alternating statistical qualification.
Acceptance and generated continuations changed. The latest C4 still takes
1.515236× reference wall time; no parity or default promotion is claimed.

The source progression adds FlashKDA prefill, optional HC prewarm, native sparse
continued prefill, BF16 TC then S8 split decode, grouped C3 MoE, batched M3 head
and within-owner MLA output. Each is explicitly gated. r24 adds bounded route
capture, not a new performance result. Current source lineage ends at
`faf4e874b2d418b67a5bc4ec5793c450c9f27ebb`; latest performance lineage is
`9b3e316a7f5ca92ec22a68414e7aefeda32023dd`.

Failed fused-MoE/HC, compact down, matrix TP, router and broader M12 screens
remain rejected. Actual-route replay passes only a routed-stage standalone gate;
it does not implement joint-owner verification or prove a C4 gain. The raw route
IDs and weight bits remain private model-derived data.

The native FP8 prefill bridge also has an explicit source-rebuild limitation:
the measured NVIDIA object is hash-pinned, but its exact source build is not
established. Do not confuse published bridge source with a reproducible upstream
object. The BF16 path remains available. [Next steps](NEXT_STEPS.md) identify this
and the remaining performance/ownership gates.
