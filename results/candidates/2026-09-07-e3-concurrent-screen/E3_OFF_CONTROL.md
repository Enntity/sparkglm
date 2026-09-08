# Identical-image E3-off control

Nine warmed repetitions use the same cap-32 image, 32-row expert threshold,
and 16384-token prefill chunks as the E3-on arm, with E3 disabled. The first
two warmup repetitions were nondeterministic in C4 and are retained. All nine
measured repetitions completed with deterministic token hashes matching the
reference.

| Arm | C1 aggregate tokens/s | C4 aggregate tokens/s | Long C2 aggregate tokens/s |
| --- | ---: | ---: | ---: |
| Reference, nine | 220.23 | 340.66 | 80.27 |
| E3 on, nine | 203.97 | 361.60 | 78.29 |
| E3 off, nine | 226.68 | 375.29 | 71.49 |

Disabling E3 recovers isolated decode in this control; the cost is not explained
solely by the new image or the expert threshold. The exact cause within E3
initialization/profiling remains unresolved. However, the E3-off control also
regresses long-prefill throughput by 10.9% against the reference and raises
median TTFT from 0.157 to 0.203 seconds. It fails the protected G2 performance
guard and does not qualify the combined 32-row/16K-chunk settings as a default.
The reference differs in chunk and threshold settings; this is not proof that
the threshold alone causes the long-prefill regression. These measurements
are separated in time and should not be treated as a pure paired kernel A/B.
