# Retained video runtime and native artifact isolation

## Question and scope

Why did the source-restored image measure slower than the retained posted-video
image? This is a diagnostic investigation, not an optimization or promotion.
The prior full-model point estimate was 5.22% lower aggregate delivered
throughput; the original consecutive-arm receipts remain unchanged in
[the restoration bundle](../2026-09-04-video-source-restoration/RESULT.md).

## Current answer

The completed [ten-repetitions-per-image comparison](TEN_REP_RESULT.md) is
effectively tied in headline speed under matched cache/CPU controls: preserved
mean wall time 88.835 seconds, rebuilt 88.907 seconds (**+0.08%**); mean
delivered throughput 23.372 versus 23.458 tok/s. The two reversed-order block
pairs change direction. The earlier 5% deficit did not reproduce.

This favors run/startup variability and prior control differences over a
repeatable missing engine optimization in this case. It does not isolate the
cause of each earlier slow run or prove equality under automatic sizing.
Runtime-environment drift remains real, and not every secondary latency metric
is identical. See the complete twenty-run table, uncertainty limits and
secondary metrics in the linked report. No default is changed or promoted.

## Artifact findings

The historical build script and source revisions are retained. The original
foundation built a full custom vLLM wheel using the manylinux aarch64 CUDA 13
builder, with KV connectors disabled. The reconstruction rebuilds the changed
native targets on the pinned upstream runtime image. These are different
build environments even after engine source restoration. A complete retained
historical builder filesystem has not been established.

The runtime dependency closure was not preserved by checking only Torch,
Triton, FlashInfer, and ExLlama versions:

| Package | Retained original | Reconstruction |
| --- | --- | --- |
| Transformers | 5.16.1 | 5.15.1 |
| Tokenizers | 0.23.1 | 0.22.2 |
| NumPy | 2.3.5 | 2.2.6 |
| nccl4py | 0.5.0 | 0.4.1 |
| Pydantic | 2.13.5 | 2.13.4 |

The reconstructed base also installs LMCache, CuPy, and other KV-connector
dependencies absent in the retained original. CuPy libraries were mapped in
the reconstructed worker, despite no configured KV connector. This establishes
environment drift, **not that these packages caused the measured slowdown**.
The complete package delta is retained in `raw/audit-summary.json`.

## Suspects ruled out or narrowed

- All four rendered prompt strings and complete token-ID sequence hashes match
  between images. Resolved vLLM model configuration differs only in its
  Transformers version field. Tokenizer version drift did not change these
  benchmark inputs.
- The separately resolved DFlash2 configuration also matches, except for its
  Transformers version field. Both resolve to `Qwen3Config`; the two complete
  configuration receipts are retained. This check covers defaults that tinyGLM,
  which has no drafter, cannot exercise.
- The inspected Torch CPU/CUDA, NCCL, cuBLAS/cuBLASLt, CUDA runtime, NVRTC,
  and FlashInfer sampling/sparse-MLA shared libraries match by file hash.
- EXL3's 769 SM121a CUDA functions have identical normalized instruction-body
  hashes. Its ELF CPU `.text` and `.rodata` sections also match. The differing
  whole-library hash is not evidence of different kernel arithmetic.
- vLLM core's 2,724 SM120 CUDA function bodies match as a multiset. 2,651 names
  match directly; 73 compiler-generated names differ with matching instruction
  bodies. Its host-side code differs, so this is not whole-binary equivalence.
- All 1,160 inspected DeepGEMM/vLLM-include/Triton source/header/configuration
  files match. This does not establish equality of every runtime-generated
  binary or its dispatch behavior.
- Both images report NVCC and PTXAS 13.0.88. Different `CUDA_VERSION` image
  environment strings are not proof of a different active CUDA compiler.
- The larger historical `_flashkda_C` library is not the GLM KDA path here:
  the inspected GLM implementation imports Triton FLA KDA operations, and
  `_flashkda_C` was not mapped in the observed reconstructed worker. Do not
  infer a missing active GLM optimization from its size alone.

## Small native timing experiment

Full GLM was unloaded. A first attempt to create a second CUDA context while
it was resident failed with out-of-memory before measuring anything; no
timings from that attempt are included as results.

The successful order was original / reconstructed / reconstructed / original,
on one GB10, CPU core 19, one OpenMP thread. Each case used five eager warmups,
32 operations per captured CUDA graph, three graph warmups, and 11 timed
replays. Peak allocated memory was 484,500,480 bytes, without model weights.
Raw per-replay timing samples are retained.

Cases cover BF16 RMSNorm at rows 1/4/32/512/7168, width 4096; and FP16 top-k
at the same row counts, score widths 16384/32768, k=512. There is no consistent
reconstruction slowdown: individual median changes range from approximately
-5.8% to +1.7%. At 7168 rows, RMSNorm is +0.55%, 16K top-k +1.71%, and 32K
top-k +0.52%. These are diagnostic samples, not a kernel-win claim or complete
G1 qualification. Matching instructions and these timings weaken the claim
that those rebuilt GPU kernels explain the full-model gap.

## Runtime isolation experiment

A private disposable hybrid image preserves the original runtime dependency
environment but substitutes the reconstructed installed vLLM and EXL3 engine
artifacts. This is an intervention to separate engine artifacts from their
runtime environment, not a source-built release or proposed default. The
historical distribution metadata remains; engine identity must use artifact
hashes rather than its package version label.

The tinyGLM arms use the same two-layer, 16-expert dummy fixture, TP2, no
speculation or vision, FP8 KV, length 32768, batch budget 7168, concurrency
limit four, and GPU-memory utilization 0.15. Each arm discards two complete
workload repetitions and retains seven. Both containers are restricted to
CPU cores 5-8 and 15-19 before the discarded warmups. This controls core class
but does not remove all unrelated CPU work. Order: reconstructed, hybrid,
original, reconstructed repeat.

| tinyGLM arm (7 retained repetitions) | C1 tok/s | C4 tok/s | long C2 tok/s |
| --- | ---: | ---: | ---: |
| Reconstructed | 284.532 | 421.412 | 87.549 |
| Hybrid: original environment, reconstructed engine | 286.865 | 423.529 | 87.759 |
| Retained original | 290.355 | 423.028 | 88.181 |
| Reconstructed repeat after restart | 291.254 | 423.046 | 87.564 |

Every output token-ID signature matches, including discarded warmups. The
second reconstructed startup exceeds the original C1 point estimate. These
data do **not** establish a stable engine-artifact or dependency-environment
penalty. They do not prove that the earlier full-model gap was noise.

An additional original/reconstructed/reconstructed/original host-launch probe
on core 19 measures a one-row RMSNorm call outside a graph and graph replay
as a control. Native-call medians were 3.656/3.704/3.704/3.720 microseconds;
graph-replay medians were 2.070/2.057/2.056/2.056 microseconds. Fifteen batches
of 1,000 calls per case were timed, excluding the final device synchronization
from host enqueue time. This focused test shows no material rebuilt host-launch
  penalty; it does not cover every C++ operator or distributed IPC path.

## Full-model replay and automatic-memory mismatch

The same production checkpoint and DFlash2 k7 video workload was replayed with
the two full images: exact original prompts, arrivals 0/1/2/3 seconds, four
400-token outputs, greedy target sampling, and unique per-run cache namespaces.
Both ranks use the same performance-core set as the tinyGLM comparison. Boot
shape warmup, long-C4 capacity warmup, and one complete video replay precede
retained runs. The probe records delivered text/timing, speculative-work counter
deltas, prefix hits, preemptions, and head-GPU clock/temperature/power samples.
Metrics-flush waits are outside the measured request wall time.

| Reconstructed retained run | Wall seconds | Aggregate delivered tok/s | Draft/request rounds |
| --- | ---: | ---: | ---: |
| 1 | 87.064 | 23.985 | 602 |
| 2 | 98.684 | 21.678 | 622 |
| 3 | 95.862 | 21.702 | 630 |
| 4 | 90.980 | 22.564 | 625 |
| 5 | 90.959 | 22.912 | 636 |
| Median | 90.980 | 22.564 | 625 |

The discarded replay was 85.851 seconds / 24.243 tok/s. It is not selected as
the result. Every retained run delivered 1,600 tokens with zero prefix hits or
preemptions. The draft counter sums per-request proposals across streams; it
is **not** a count of batched GPU forward passes. First-token and termination
accounting, including speculative overshoot, mean accepted-plus-bonus counters
need not equal delivered tokens exactly.

The first request's TTFT varies from 20.247 to 25.057 seconds. Consequently,
draft acceptance alone is not an explanation of all observed variation.
The retained-original automatic-cache comparison completed five measured
repetitions: wall seconds 89.508 / 91.341 / 89.644 / 89.616 / 86.417 and
aggregate delivered tok/s 23.081 / 22.681 / 22.997 / 23.283 / 24.358.
Its medians are 89.616 seconds and 23.081 tok/s. The rebuilt medians are
therefore 1.52% longer wall time and 2.24% lower throughput, smaller than the
earlier 5.22% throughput point estimate. These consecutive arms still do not
separate startup, memory allocation and within-run variation from image effects.

Identical flags did not freeze the realized KV-cache budget:

| Head-rank startup measurement | Reconstruction | Retained original |
| --- | ---: | ---: |
| Memory-utilization setting | 0.87 | 0.87 |
| Reported consumed memory | 82.07 GiB | 84.05 GiB |
| Reported activation headroom | 4.46 GiB | 5.38 GiB |
| Reported graph memory | 0.47 GiB | 0.59 GiB |
| Auto-selected KV budget | 19.34 GiB | 16.44 GiB |

The historical `vllm/utils/mem_utils.py` deliberately substitutes
`psutil.virtual_memory().available` for CUDA free-memory reporting on an
integrated GPU. It then computes consumed memory from the difference between
the before-create and after-profile snapshots. `gpu_worker.py` subtracts that
consumption and transient headroom from the requested memory budget. Thus the
result depends on system RAM state and allocation behavior during startup,
not just the flag or weight hashes. A concurrent host-RAM change can enter
the consumption estimate. This explains why equal utilization flags are not
an exact memory-budget control; it does **not** prove that the budget delta
caused a particular throughput delta.

The applicable source is the retained foundation
`6a49e49e7e6a3226197a2ceefcf217cdf55f751e`; this logic is also present in the
source-restored engine. The original actually allocated 650 blocks / 1,132,404
reported cache tokens. The rebuilt automatic-cache arm reported 1,168,989
tokens. Head-rank estimated budgets in the table are not the actual paired
allocation: TP2 is constrained by the limiting rank. They must not be read as
a realized 2.9 GiB pool difference.

A completed stronger ten-repetition comparison uses
`--num-gpu-blocks-override 650` on both images and verifies actual reported
pool geometry. Its order, metrics, warmup and interpretation limits were
declared in [the ten-repetition protocol](TEN_REP_PROTOCOL.md) before retained
fixed-cache results were inspected. No such setting has been promoted or
substituted for the video default here.

The loaded OS-library audit separately covers 254 non-package shared-library
paths. There are 36 different file hashes, mostly Python extension and system
package builds. Loaded libc, libstdc++, libibverbs, and libmlx5 match byte for
byte. No missing RDMA-provider library was found. The hybrid experiment also
preserved this original OS environment; its tinyGLM result was not a stable win.

## Limitations

The focused twenty-run full-model comparison is complete; it is not the full
G3 workload matrix, and no default change is implied.
tinyGLM cannot reproduce full-depth numerical drift, DFlash acceptance, or
full-model bandwidth pressure. Identical instructions do not prove identical
launch order, host overhead, cache state, or selected runtime paths. Until a
controlled intervention reproduces the gap, its cause remains unproven.

Probe scripts are original diagnostic orchestration using Python standard
libraries, PyTorch CUDA events/graphs, and NVIDIA binary-inspection tools.
No upstream optimization or GPU binary is introduced or redistributed here.
