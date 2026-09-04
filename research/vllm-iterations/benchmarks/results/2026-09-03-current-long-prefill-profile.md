# Current accepted long-prefill profile

Date: 2026-09-03

## Verdict

The current accepted paired/fused E2 build is still dominated by routed EXL3
MoE compute during long prefill. This synchronized profile supersedes the old
pre-E2 attribution used to order the remaining work.

The next large-lever experiment must optimize the expert compute itself. More
MoE dispatch rearrangement and fused sparse score/top-k work are not justified
by this profile.

## Exact artifact and workload

- Source commit: `eba60e7048` (`perf(glm): fuse exact EXL3 fat expert pipeline`)
- Live image label revision: `6a49e49e7e6a3226197a2ceefcf217cdf55f751e`
- Live `exl3.py` SHA-256:
  `edf9a35071c6c20980b4aa41af6275b3b985400abbab4db8592f9b7395ff3083`
- TP2 over the two GB10s, EXL3 K4 MCG target, DFlash2 K7/TP2, FP8 KV
- `max_num_batched_tokens=7168`, `max_num_seqs=4`
- One 32,176-token C1 isolation prompt
- Torch profiler enabled on both ranks with no stacks, no shape recording and
  no memory recording
- Three short requests calibrated the 12-worker-step delay. The accepted
  profile contains four `execute_context_1(7104)` scopes and two complete
  long-prefill target forwards (84 MoE calls, 68 KDA calls and 22 sparse-MLA
  calls).

The profiled request completed 8/8 requested output tokens without an API
error. Profiling overhead raised TTFT to 35.16 seconds, so its endpoint timing
is not a throughput result.

## Unprofiled anchor

The same restored service was measured immediately before the profiler-only
restart.

| Case | Aggregate prefill | Aggregate decode | Wall | Correctness |
| --- | ---: | ---: | ---: | --- |
| Warm medium C2 | 1,332.65 tok/s | 29.05 tok/s | 29.60 s | 2/2 complete; markers isolated |
| Large C2 | 1,313.91 tok/s | 11.22 tok/s | 54.44 s | 2/2 complete; markers isolated |

The large row reproduces the accepted change receipt's 1,313.9 tok/s result.

## Rank-symmetric CUDA attribution

PyTorch's table duplicates CUDA time into enclosing CPU scopes, so percentages
below use either a named leaf kernel or a named operation boundary against each
rank's 10.22-10.24 seconds of leaf CUDA time. Enclosing totals must not be
summed with their child kernels.

| Operation | Rank 0 | Rank 1 | Meaning |
| --- | ---: | ---: | --- |
| `vllm::moe_forward_shared` CUDA total | 5.037 s | 4.909 s | 49.3% / 48.0% of the captured CUDA time |
| Fat gate/up direct trellis kernel | 1.448 s | 1.397 s | 14.2% / 13.7% |
| Fat down/scatter direct trellis kernel | 1.142 s | 1.139 s | 11.2% / 11.1% |
| Base `exl3_moe_kernel` | 1.145 s | 1.125 s | 11.2% / 11.0% |
| TP all-reduce kernel | 1.041 s | 1.224 s | 10.2% / 12.0% |
| mHC fused post/pre boundary | 0.888 s | 0.898 s | 8.7% / 8.8% |
| Sparse MLA attention boundary | 0.906 s | 0.878 s | 8.9% / 8.6% |
| Sparse MLA compute kernel | 0.583 s | 0.563 s | 5.7% / 5.5% |
| Full sparse indexer boundary | 0.056 s | 0.055 s | 0.55% / 0.54% |
| FP8 sparse scorer kernel | 0.0156 s | 0.0152 s | about 0.15% |
| Exact FP16 top-512 kernel | 0.0061 s | 0.0062 s | about 0.06% |
| MoE grouped router top-k | 0.0061 s | 0.0061 s | about 0.06% |

The fat path issued 6,589 gate/up kernels, 6,589 down/scatter kernels and
6,589 fused SwiGLU/Hadamard kernels in the captured rank-0 interval. The E2
diagnostics were identical on both ranks and showed no batched, sorted or
legacy fallback.

KDA remains material in aggregate because 34 layers each run the multi-stage
chunk pipeline. Its individually named stages include roughly 0.15 seconds
each for output formation and recomputation, 0.12 seconds each for the main
chunk kernel and short convolution, and additional triangular/intermediate
kernels. It is secondary to MoE but remains a legitimate later fusion target.

## Consequences

1. Optimize the direct EXL3 expert kernels first. The paired gate/up and
   down/scatter kernels alone are about one quarter of total CUDA time, and the
   complete MoE boundary is about one half.
2. Profile and optimize exact TP2 reductions second. They are already a
   double-digit share and differ noticeably by rank.
3. KDA fusion remains third.
4. Optimize the sparse MLA attention kernel/layout, not the scorer/top-k. The
   full indexer is under one percent in the measured current engine; even a
   free selector cannot create a meaningful endpoint win.
5. Do not revisit host-dispatch grouping without new evidence. The accepted
   path's production-shape microbench already showed the grouped E2 design was
   neutral to negative.

## Raw artifact identity

The full traces remain outside git because each is about 116 MiB. Their
temporary analysis location is
`/private/tmp/sparkglm-profile-accepted-eba60e7048-20260903a/`.

- rank-0 trace SHA-256:
  `3180b36177f902455b691f6e9fa07bf0106e20a3690a62e195fe1929c3b53542`
- rank-1 trace SHA-256:
  `fc27896d4f82ce9144fbdceed1e73685bae763df0ccbd1eef55597b4490b4fe0`
- rank-0 table SHA-256:
  `1b76e4bd8a4e91adfee54750645739ccdd9c3723509340887d1f249dec359320`
- rank-1 table SHA-256:
  `f12e3b0dba92e6965f85511a4192c526cc7538b83e753f35d482f543b4608a34`
