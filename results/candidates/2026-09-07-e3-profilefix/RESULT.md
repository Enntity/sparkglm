# E3 corrected profiling integration

| Arm | C1 aggregate tokens/s | C4 aggregate tokens/s | Long C2 aggregate tokens/s |
| --- | ---: | ---: | ---: |
| reference | 232.72 | 386.23 | 78.72 |
| profilefix | 248.92 | 383.78 | 79.19 |

All measured streams complete, are deterministic, and match the reference token hashes. The automatic checks in checks.json also protect median wall time and TTFT. Warmup receipts, including any initial nondeterminism, are retained and excluded from these medians.

The fixed-budget control uses the earlier adapter. The profiling-fix control uses the corrected adapter over the original non-MXFP8 base: it reserves E3 buffers then executes the reference profiling path, accounting for both persistent scratch sets. These results do not prove that cache sizing alone explains every earlier timing change; expert thresholds and allocation layout also differ.

This is G2 integration evidence, not full-model speed or quality qualification. E3 remains opt-in pending those gates.
