# EXL3 K4 codebook LUT: rejected

Date: 2026-09-03

## Verdict

Reject before model integration. Replacing the K4/MCG procedural codebook
reconstruction with an exact 65,536-entry FP16 table is about **9.8-10.9x
slower** for the complete routed-expert body on one GB10. The table is only
128 KiB and the result is bit-identical, but divergent per-lane lookup traffic
serializes far more severely than the integer reconstruction dependency chain.

The experimental extension entry points were removed after the result. The
benchmark harness is retained as `benchmarks/kernels/benchmark_exl3_k4_lut.py`
and documents the rejected interface and complete test boundary.

## Isolated change

Both implementations lived in one extension binary and used the accepted
M64/K16/N128 geometry, three-stage `cp.async` pipeline, packed K4 weights,
FP32 accumulation, exact rounded SwiGLU, Hadamard/scales, route weighting and
scatter. The candidate changed only `decode_3inst<1>`:

- control: eight procedural reconstructions per lane from the exact sliding
  16-bit windows;
- candidate: the same windows indexing a device-resident 128-KiB FP16 table
  initialized by `decode_3inst<1>` itself.

No model weights were loaded. Random packed tensors used the GLM TP-local
hidden/intermediate dimensions of 4,096 and 1,024. Timings used 10 warmups, 40
iterations and five alternating-order repeats.

## Complete expert-body result

| Expert rows | Procedural M64 | LUT M64 | LUT slowdown |
| ---: | ---: | ---: | ---: |
| 129 | 0.1131 ms | 1.1150 ms | 9.86x |
| 320 | 0.1875 ms | 1.9434 ms | 10.36x |
| 512 | 0.2819 ms | 3.0667 ms | 10.88x |
| 1,024 | 0.5613 ms | 6.0894 ms | 10.85x |
| 2,048 | 1.1896 ms | 11.8013 ms | 9.92x |
| 4,096 | 2.3462 ms | 23.7435 ms | 10.12x |
| 6,528 | 3.8615 ms | 37.8326 ms | 9.80x |

The paired gate/up LUT kernel alone was 10.99-15.96x slower. The down/scatter
LUT kernel was 6.55-8.93x slower. Every final output had `max_abs=0` and
relative RMSE 0.

## Consequence

The EXL3 procedural codebook is not replaceable by ordinary cached lookup on
SM121, even when the complete table is tiny relative to L2. Preserve its
compact arithmetic and target overlap instead: decode a future packed-weight
fragment while tensor-core work consumes the current fragment. A useful next
candidate must change instruction scheduling or producer/consumer dataflow,
not add decoded-weight memory traffic.
