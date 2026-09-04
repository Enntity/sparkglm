# SM121 EXL3 fat-GEMM `cp.async` acceptance — 2026-09-02

## Verdict

Accept for the opinionated GLM-5.3 Flash two-Spark appliance.

The candidate replaces the synchronous global-to-shared loads and two
barriers per K block in the EXL3 fat-expert GEMM with Reederey's three-stage
`cp.async` pipeline. The final valid image is an exact derivative of the
accepted native-FP16-indexer image: the EXL3 extension is the only executable
artifact replaced.

On the fixed 16K/32K staggered C1/C2 workload, every prefill metric improved.
C2 wall time fell by 6.66% at 16K and 7.55% at 32K. All requests completed,
retained their own isolation marker, and contained no marker from a peer.

## Source and attribution

- Source commit:
  `Reederey87/glm53-flash-exl3-2x-dgx-spark@0c03250cd7176a2fef9cbbf9329fed08c8750e7d`
- Original author: Reederey (Artem Matskevych)
- Original co-author: `factory-droid[bot]`
- Local source SHA-256:
  `47d8dec4ad6a4ac9bbf4f351ffa5762d044baf4f02cdfdc9771e49a1764c7ec0`
- The local and source-repository CUDA files were byte-identical before build.

The implementation uses the `cp_async`, `cp_async_fence`, and
`cp_async_wait` helpers already present in the pinned exllamav3 source. The
three-stage fat-GEMM pipeline itself is Reederey's change.

## Frozen artifact and configuration

- Control image: `sparkglm-vllm:fp16-indexer-candidate`
- Control image ID:
  `sha256:107d899515629f168fdd5f7bac3e967bc3d0fe768124bafd9322c7a3a8c4adc4`
- Candidate image: `sparkglm-vllm:fp16-cpasync-0c03250`
- Candidate image ID:
  `sha256:168d27a9abf37ce19cfe5566dbf6139b55dbcda6b15032746c02feed89a6f528`
- Inherited vLLM source revision:
  `6a49e49e7e6a3226197a2ceefcf217cdf55f751e`
- Hardware: two DGX Spark GB10 systems, TP2 over ConnectX-7
- Model: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- KV cache: FP8
- Scheduler: 7,168-token budget, work-conserving mixed prefill
- Speculation: DFlash2, seven draft tokens, draft TP2
- Spin wait: 16 ms
- Workload: 128 forced output tokens; C2 starts separated by five seconds
- Paired prompt salt: `cpasync-pair-20260902`

Both ranks reported the same candidate image ID. Startup reported native FP16
indexer support, `kernel_ok`, all 42 profiled prefill layers on the fat path,
and no fallback.

## Paired same-prompt real-workload A/B

Positive percentages mean the candidate is faster. Prompt SHA-256 values
matched for every control/candidate request.

| Case | Candidate/control TTFT (s) | TTFT delta | Candidate/control aggregate prefill (tok/s) | Prefill delta | Candidate/control wall (s) | Wall delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16K C1 | 13.14 / 15.27 | +13.99% | 1203.3 / 1034.9 | +16.26% | 18.12 / 20.93 | +13.43% |
| 16K C2 | 22.70, 20.71 / 24.36, 22.69 | +6.84%, +8.74% | 1229.8 / 1141.3 | +7.75% | 32.55 / 34.87 | +6.66% |
| 32K C1 | 25.89 / 27.65 | +6.37% | 1243.0 / 1163.8 | +6.81% | 31.14 / 32.62 | +4.55% |
| 32K C2 | 33.85, 46.70 / 36.25, 50.83 | +6.61%, +8.13% | 1244.9 / 1152.7 | +7.99% | 57.38 / 62.07 | +7.55% |

Aggregate delivered decode throughput also improved in the mixed C2 cases:
24.18 to 25.82 tok/s at 16K (+6.76%) and 9.84 to 10.80 tok/s at 32K
(+9.76%). Active per-request decode rates remain output- and scheduling-noisy;
the kernel itself is prefill-only.

## Isolated GB10 kernel A/B

`benchmarks/kernels/benchmark_exl3_fat_gemm.py` times the installed extension
without loading model weights. These direct-epilogue runs used M=3584,
N=2048, 20 warmups, and 100 timed iterations on the rank-0 DGX Spark.

| K | Control mean (ms) | Candidate mean (ms) | Control/candidate equivalent TFLOP/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.6653 | 0.5050 | 45.19 / 59.53 | +31.74% |
| 4096 | 1.2537 | 0.9126 | 47.96 / 65.89 | +37.37% |
| 8192 | 2.4374 | 1.7477 | 49.34 / 68.81 | +39.47% |

## Correctness and quality boundary

The candidate passed the existing GPU EXL3 suite on GB10:

- SM121a cubin and direct/scatter symbols present.
- Direct and scatter epilogues finite and within the reference bound.
- Fused, mixed thin/fat, diagnostic-counter, expert-map, and CUDA-graph gates
  passed.
- All twelve requests in the paired real-workload matrix completed 128/128
  tokens with `finish_reason=length`, preserved their own marker, included no
  foreign marker, and returned no error.

The upstream source commit additionally reports 56 bit-exact comparisons and
clean compute-sanitizer memcheck, racecheck, and synccheck runs. Those
sanitizer claims were not independently rerun here; our local qualification
used the repository's GPU parity suite and end-to-end isolation gate.

## Discarded experiment

An initial candidate was built directly from Mia's public base rather than
from `sparkglm-vllm:fp16-indexer-candidate`. It lacked the local
`indexer_logits_dtype` implementation and therefore changed both the vLLM core
and the EXL3 kernel. Every full-engine result from that image was discarded.

The final image was built `FROM sparkglm-vllm:fp16-indexer-candidate`, then
only the pinned exllamav3 extension was rebuilt with the byte-identical
`0c03250` CUDA source. Its inherited source labels and native FP16 selector
were checked before the valid model load.

## Capacity note

Candidate startup reported 1,331,010 KV tokens versus 1,339,721 in one earlier
startup. This is a 0.65% difference and does not constrain the 16K/32K target.
It is not attributed to the kernel: dynamic shared memory is per resident CUDA
block, not persistent global KV storage, and vLLM's startup activation/graph
profiling varies between boots.
