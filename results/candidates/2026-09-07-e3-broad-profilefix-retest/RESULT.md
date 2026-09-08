# Corrected broad E3 policy retest

The cheap screen does not justify another full-checkpoint load. Broad dispatch with the corrected profiling adapter, 32 expert temporary rows, and 7168-token chunks preserves the reference token hashes but fails the tinyGLM throughput guard.

| Case | Reference tokens/s (n=9) | Broad E3 tokens/s (n=3) |
| --- | ---: | ---: |
| decode_c1 | 232.72 | 239.83 |
| mixed_c4 | 386.23 | 382.91 |
| long_c2 | 78.72 | 72.57 |

The long-case samples are retained individually, including the faster first sample. This is a bounded screening decision, not a claim that broad E3 can never win a different full-model workload. The passing concurrent-only profile remains the EXL3 finalist. No full-model run was performed for this broad-policy retest.
