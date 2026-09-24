# Atlas native source rebuild

The source-built sparse MLA library matched the retained runtime output and LSE bytes on four fixtures (2048, 3515, 4100 and 4096 rows). Prepared Q/KV buffers matched an independent Torch construction. All input hashes, buffer guards and 42 invalid-argument checks per fixture passed. Peak Torch reservation was 2,019,557,376 bytes, below the 2 GiB test cap.

The source-built FlashKDA library and precise adapter matched output and FP32 recurrent-state hashes on six fixtures: 2048, 4096 and 4100 tokens, each with ordinary and degenerate inputs. Each implementation ran in a separate process to avoid shared-library aliasing. Inputs, guards and rejected-capacity checks passed.

The initial sparse rebuild used different arithmetic flags and failed exact parity. Matching FlashInfer's pinned JIT flags resolved it. A later test reached its own 2 GiB allocation cap; Q reference checks were chunked while preserving every fixture and the cap. Neither rejected run changed serving state.

These are synthetic replacement checks, not full-model quality, throughput, context-length or endurance qualification. The historical cached object still has unresolved source identity; the replacement has its own pinned source and different binary hash.
