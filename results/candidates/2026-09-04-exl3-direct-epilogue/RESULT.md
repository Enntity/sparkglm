# Direct grouped EXL3 epilogue: rejected operator screen

## Decision

Neither variant advances to tinyGLM or a full-model run. All completed
arithmetic and graph checks passed, but neither met the predeclared performance
screen. No serving default, checkpoint, or public baseline changed.

| Comparison | Active-shape geometric mean latency change | Decision |
| --- | ---: | --- |
| Direct epilogue vs preserved reference | +5.88% (slower) | Reject |
| Direct epilogue vs same-toolchain rebuilt reference | +6.02% (slower) | Reject |
| Contiguous-atomic follow-up vs rebuilt reference | +2.39% (slower) | Reject |

Each comparison contains three alternating-order pairs, three matching seeds,
14 shapes, 10 warmups and seven samples of 40 CUDA graph replays per shape.
The cap=128 no-work case is excluded from the active-shape geometric mean.
Every retained sample is included; isolated wins are not selected as gains.
The screen required at least 3% lower latency and no reproducible shape
regression above 2%. The follow-up's two mixed cases regress about 6.0% and 6.6%.
These are **operator latency measurements, not end-to-end token rates**.

## Hypothesis and diagnostic

Keep the output Hadamard in registers and write directly, removing a shared
write/read and one epilogue barrier. The standalone M64 oracle, trellis
arithmetic, MMA loop, and grouped planner remain unchanged.

A separate, single diagnostic profile of the eight-fat-expert case showed the
direct variant's down projection at 746 us versus 620 us in the rebuilt
reference; gate/up was approximately unchanged (1137 vs 1129 us).
Those profiler-perturbed means are diagnostic only, not another speed test.
Compiled resource inspection reported 64 registers for both gate/down kernels,
with stack frames increasing from 8 to 16 bytes in the direct variant. That
does not establish an occupancy change or prove a spill-based cause.

The direct variant issues each scalar atomic at stride four across a warp.
The follow-up uses warp shuffles to restore contiguous destination addresses
for each atomic instruction. It reduced the overall regression, but did not
beat the reference and regressed both mixed workloads. This supports an output
schedule cost, but does not establish a complete hardware-level explanation.
Do not assume fewer shared-memory operations imply faster execution.

## Identity and reproducibility

Published source control: `fef5152ef20159b33c461989284942620d077037`.
Initial direct implementation: `38bd86adb3418827af98e993f20deea8af56d27e`.
Build context began at `b650255` with local incremental-build and harness edits;
the follow-up was also tested before commit. Receipts bind actual CUDA-source,
extension-binary, image, and harness hashes. These are not clean public binaries.

The initial candidate compiled in about 110 seconds. Reference and follow-up
recompiles took about 30 seconds each by retaining the pinned ExLlamaV3 object
tree. No vLLM native rebuild, checkpoint download, or model load was required.

Use the source-locked build/runner in the
[experiment guide](../../../research/experiments/exl3-direct-epilogue/README.md).
Reproduce each table row with `summarize.py` and its corresponding `raw/`
directory: `preserved-control`, `rebuilt-control`, or `coalesced`.
The unchanged reference was rebuilt in the candidate toolchain specifically
to rule out a preserved-versus-new-extension build confound.

## Correctness and harness corrections

All 18 completed arm runs passed exact gather/Hadamard, gate/up, and activation
comparisons; single-expert output was bit-exact. Overlapping-expert FP32 atomic
sums used the predeclared `rtol=1e-5, atol=1e-6`. Output red zones and repeated
CUDA graph replay passed. Inputs are random dummy trellises at GLM TP2 geometry.

Two earlier reference-only attempts stopped on harness defects: the all-thin
fixture violated `cap < tokens`, and the oracle indexed compact fat-only scratch
using uncompressed route offsets. Both were fixed, with CPU regressions added,
before any retained comparison. No tolerances were loosened. Failed attempts
remain outside this receipt set and are not counted as model/kernel failures.

## Limitations and next direction

No sanitizer, tinyGLM, full-model, semantic, or endurance qualification. The
failed performance screen made further qualification unjustified. No GPU
clocks were locked; the three-pair screens characterize this run, not a
confidence-bounded sub-percent effect. Warm synthetic weights do not reproduce
full-checkpoint cache pressure, actual routing frequencies, or TP2 communication.
This used one reserved GB10 for a TP2-local operator, not two-rank serving.

The next investigation should address the grouped trellis/MMA main loop and
its synchronization rather than assuming an epilogue-only rewrite is a large
lever. That is a direction for measurement, not a prepared or measured gain.

## Provenance

Original SparkGLM scheduling, source-locked variants, build-cache workflow and
tests. Hadamard/trellis arithmetic retains ExLlamaV3 provenance at
`c5d9c657966ffeeaa9353f0cc899f18629da4a13`; the M64 pipeline retains Reederey
provenance at `0c03250cd7176a2fef9cbbf9329fed08c8750e7d`. Incremental build
setup adapts the existing Mia-derived root recipe. No upstream direct-epilogue
or contiguous-atomic implementation was copied; no checkpoint weights were used.
