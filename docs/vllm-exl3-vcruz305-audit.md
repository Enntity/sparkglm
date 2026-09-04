# `vcruz305/vllm-exl3` applicability audit

Date: 2026-09-03

Target under evaluation: GLM-5.3-Flash EXL3 on two DGX Sparks, TP2, K4
routing, concurrent and long-context serving.

Upstream revisions inspected:

- `vcruz305/vllm-exl3`: `67dc742`
- its pinned ExLlamaV3: `17bc3923259ffd48aab742edd261a0ca45d55459`
- our pinned ExLlamaV3: `c5d9c657966ffeeaa9353f0cc899f18629da4a13`

## Decision

Do not vendor or install this implementation. It is a useful single-Spark K2
prototype, not a qualified implementation for our TP2/K4 target. Reimplement
the promising kernel ideas against our own ABI, semantics, batching, graph, and
correctness requirements.

## Why the published result is not comparable

The published GLM recipe is TP1, K2, and `MAX_NUM_SEQS=1`. Its two physical
Sparks are used as independent baseline and candidate machines, not as a TP2
pair. This does not test TP communication, K4 routing, concurrent rows,
staggered admission, mixed prefill/decode, or 16K-32K prefill fairness.

The recipe's `--long-prefill-token-threshold 1024` setting therefore cannot be
used as evidence for concurrent scheduling behavior: with one admitted
sequence, there is no competing decode stream to protect.

## Source-level findings

### Native fused MoE path

The native dispatch guard accepts only 1-8 input rows, hidden size 4096, K2/K3/K4,
and a local intermediate size of exactly 2048. Our TP2 GLM routed experts are
sharded rank-locally and do not present that exact local shape, so this path
cannot dispatch for our serving configuration without redesign.

The Python wrapper loops over batch rows and launches one fused call per row.
The C++/CUDA wrapper allocates gate, up, down, Hadamard, and accumulator tensors
on each call, including a zero-filled accumulator. That is incompatible with
the persistent scratch, batched launch, and CUDA-graph-safe path we want.

The path also downcasts routing weights/output to FP16 and does not implement
the GLM SwiGLU clamp-limit semantics already handled by our current path.

Its test coverage is a single synthetic K2 row and accepts cosine similarity
of 0.999. It does not cover a real GLM checkpoint, exact token/logit agreement,
K4, TP2, DFlash, CUDA graph replay, concurrency, or long context.

### Prefill path

The advertised `exl3_gemm` implementation chunks the row dimension into groups
of at most eight and repeatedly invokes GEMV plus copies. That is not a native
large-M GEMM implementation.

The subsequently added `exl3_fat_gemm` source is attributed upstream as copied
from Mia's `4b8d3c7` implementation with include-only edits. Its fast dispatch
is guarded for K4. We already carry that lineage and have subsequently added
our own M64, paired/fused, and asynchronous-copy work, so there is no new
prefill kernel to port from this repository.

### Packaging and reproducibility

The source tree identifies itself as v0.3.1 and builds a separate native
extension named `vllm_exl3_c`, but the linked Hugging Face runtime currently
offers the Python `vllm_exl3` 0.2.1 wheel. The prebuilt recipe installs that
wheel, while `verify_runtime.py` verifies the stock
`exllamav3_ext.exl3_moe` symbol rather than the new
`vllm_exl3_c.p2b_fused_moe` symbol. The documented prebuilt path therefore
does not establish that the claimed native extension is present or active.

## Controlled compatibility probe

On Spark 01, the extension was compiled in an isolated temporary container
against the exact ExLlamaV3 headers used by our live image. Compilation failed
because the source expects the older `exl3_gemv_ns` API and
`had_hf_r_128_inner` symbol, neither of which is present in our pinned ABI.

A second build against the author's pinned headers was stopped before
completion because it saturated host CPU/SSH for several minutes while the
static dispatcher and semantic checks had already disqualified a direct port.
The temporary build container was removed. The live rank-0 serving container
was not restarted and its health endpoint remained healthy.

## Ideas worth carrying forward

1. Build our own cooperative top-8 routed-expert decode kernel for TP2/K4. It
   should process all active rows in one launch, use persistent preallocated
   scratch, preserve FP32 router weights/accumulation and GLM clamp semantics,
   and support CUDA graph replay.
2. Evaluate an EXL3 overlay for the non-routed/dense linear layers as a separate
   checkpoint experiment. The upstream single-Spark result makes this idea
   interesting, but it changes storage, loading, memory use, and numerical
   behavior and must pass our real quality suite and TP2 tests.
3. Treat the long-prefill threshold as a cheap scheduler sweep only. Test it in
   the warmed C1/C2 staggered 16K-32K harness; do not adopt the one-sequence
   setting as a conclusion.

## Required acceptance gate for any extracted implementation

- Builds against the same pinned source/headers as the serving image.
- Dispatch is proven active on both TP2 ranks for GLM K4.
- Exact endpoint token agreement and bounded logit error on a fixed corpus.
- CUDA graph capture/replay and persistent-allocation checks.
- Warm, repeated C1/C2 staggered 16K-32K prefill/decode benchmark.
- No regression in TTFT, inter-token latency, aggregate throughput, or rank
  synchronization.
- Separate commit only after the measured target is met.

## Implemented experiment and tinyGLM result

The first item above is now implemented as an opt-in candidate rather than a
port of the upstream wrapper. `exl3_decode_moe_k4` is fixed to the appliance's
SM121 / TP2-local K4 shape (H=4096, I=1024, top-8). It uses one cooperative
launch for the routed layer, expert-sorted tiles of up to eight rows, persistent
caller-owned scratch, exact GLM clamp/SwiGLU semantics, FP32 routed
accumulation, and CUDA-graph capture/replay. The dispatch remains off unless
`EXL3_DECODE_COOP_K4=1`; its measured useful range is 1-16 tokens, so 16 is the
default cutoff and wider batches use the prior kernel.

On 2026-09-03 the exact-shape direct A/B passed numerical parity over 100 timed
samples per shape. Median old-to-new kernel latency was:

| tokens | old ms | new ms | improvement |
|---:|---:|---:|---:|
| 1 | 0.4107 | 0.3607 | 13.9% |
| 2 | 0.5974 | 0.5449 | 8.8% |
| 4 | 0.6051 | 0.5719 | 5.5% |
| 8 | 0.6138 | 0.5757 | 6.2% |
| 16 | 0.6247 | 0.5862 | 6.2% |
| 32 | 0.6521 | 0.6518 | fallback parity |

A reverse-order, 15-repetition, fully warmed two-Spark tinyGLM endpoint A/B
used the same image with only the dispatch switch changed:

| case | baseline tok/s | candidate tok/s | delta | token IDs |
|---|---:|---:|---:|---|
| decode C1 (128 in / 256 out) | 276.4 | 291.6 | +5.5% | exact |
| staggered C4 (4096 in / 128 out) | 412.0 | 423.9 | +2.9% | exact |
| long C2 (16384 in / 32 out) | 86.7 | 86.5 | -0.2% | exact |

This clears the tinyGLM gate and demonstrates a decode-loop gain. It does not
yet clear the real-checkpoint acceptance gate: the full 45-layer model must
still prove exact endpoint quality, rank behavior, C1/C2 decode throughput, and
16K-32K staggered TTFT before this switch becomes a production default.

## Sources

- <https://github.com/vcruz305/vllm-exl3>
- <https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe>
- <https://huggingface.co/vcruz305/GLM-5.3-Flash-EXL3-K2-spark-vllm>
