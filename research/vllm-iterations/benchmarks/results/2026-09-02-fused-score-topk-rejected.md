# Fused sparse scoring plus online top-k: rejected cluster prototype

Date: 2026-09-02

## Hypothesis

Fuse GLM-5.3's FP8 sparse-index score calculation with exact top-512 pool
selection on SM120, so the complete FP16 `[query, eligible_pool]` score matrix
is never written to and read from GPU memory.

This is the right operation boundary in principle. The experiment tested
whether an eight-CTA cluster could make that boundary fast on GB10 despite its
99 KiB per-CTA shared-memory limit.

## Prototype

- Eight clustered CTAs cooperatively score one four-query tile.
- Each CTA retains 1,024 FP16 scores per query in shared memory.
- Two distributed-shared-memory radix passes find the exact FP16 threshold.
- The cluster writes only 512 pool indices per query.
- The kernel is specialized to GLM's 32 index heads, head dimension 128,
  four-query tile, four-token K pools, top-512 pools, and SM120.

The first scoring build exposed a 512-byte shared-memory layout bug: placing a
512-byte weight tile immediately before the K tile violated the K tile's 1 KiB
swizzle alignment. Padding the weight allocation to 1 KiB made all 2,048 scores
in the 4x512 isolation probe bit-exact with the current DeepGEMM scorer.

After that correction, all production-size probes selected exactly the same
FP16 value multiset as the current materialized scorer plus vLLM selector.
Different index rows reflect only equal-FP16 tie ordering.

## Result

All times are warm CUDA-event p50 measurements on one GB10 using the exact
container and kernels from the accepted FP16-indexer candidate. `baseline` is
DeepGEMM FP8 scoring to FP16 logits followed by
`top_k_per_row_prefill`. `fused` is the exact cluster prototype.

| Rows | Pools | Score matrix | Scorer | Selector | Baseline pair | Fused | Fused / baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4,096 | 4 MiB | 125.2 us | 26.1 us | 146.2 us | 2,880.0 us | 19.7x slower |
| 512 | 8,192 | 8 MiB | 231.4 us | 34.4 us | 261.2 us | 3,000.6 us | 11.5x slower |
| 2,048 | 8,192 | 32 MiB | 824.7 us | 155.2 us | 969.4 us | 12,016.3 us | 12.4x slower |

The existing selector is only 13-16% of the measured score-plus-select pair.
At 512x8,192 it scans 8 MiB in about 34 us, already close to GB10's useful
memory-bandwidth floor. Eliminating the selector completely would therefore
cap the microkernel-pair gain at roughly 13-16%; an implementation which still
rescans logits cannot realize even that ceiling.

## Decision

Do not integrate or ship the cluster kernel. It is exact, but distributed CTA
synchronization and the eightfold block count overwhelm the avoided global
score traffic.

Also reject the simpler "score while accumulating a histogram, then rescan"
variant as the next implementation: the current FP16 selector already performs
one efficient matrix scan, so that variant preserves both the global score
write and the global score read while adding histogram atomics to the scorer.

The remaining ways to remove the matrix require a different algorithmic
decomposition, not another scheduler flag:

1. a single-CTA exact selector with a compact candidate representation that
   fits alongside the winning pipelined scorer in 99 KiB;
2. a quality-gated approximate threshold/candidate scheme that emits far fewer
   than all scores, followed by exact selection over those candidates; or
3. a model-level change to selection semantics.

Any new candidate must first beat the isolated scorer-plus-selector numbers
above, preserve exact selected FP16 values (unless explicitly evaluated as an
approximation), and only then earn an end-to-end 16K/32K C1-C2 test.
