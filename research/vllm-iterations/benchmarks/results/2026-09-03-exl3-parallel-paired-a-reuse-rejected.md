# EXL3 parallel paired gate/up A-tile reuse: rejected

Date: 2026-09-03

## Verdict

Reject the 512-thread, two-warp-group gate/up experiment. It preserves the
accepted kernel's 64-register working set and is bit-exact, but it is slower at
every tested row count. The candidate was never wired into serving.

This closes gate/up CTA co-residency as the next optimization direction on
GB10. Future work should keep the accepted independent 256-thread CTA
scheduling and optimize work within each matrix.

## Candidate

The earlier serial paired-A experiment halved the CTA count but increased
register use from 64 to 98 per thread. This second design tested whether the
same A-load reuse could win while preserving parallel execution:

- one 512-thread CTA per corresponding 128-column gate/up tile pair;
- eight warps independently compute gate and eight independently compute up;
- one shared three-stage A pipeline;
- separate packed-B and output-scratch banks;
- unchanged K4 MCG decode, FP32 accumulation, scaling and output order.

The compiled `sm_121a` cubin reports 64 registers per thread for both the
candidate and accepted M64 paired kernels. The experiment therefore isolates
the larger CTA/block-scheduling and shared-dataflow trade from the previous
register-pressure failure.

- Source base: `27cb29aa75`
- Candidate image: `sparkglm-vllm:exl3-pairparallel-candidate-20260903a`
- Head image ID: `76e0a9d16d8bb0e40849b632033ac6af84d5125dba1030a55310f88e3fd40461`
- Worker image ID: `e4d024de704b8f8a5fa71cf958f0530a1eb64e5a2ba5d13f066d0ad76b41849b`
- Shape: hidden 4,096, TP-local intermediate 1,024
- Timing: CUDA events, 20 warmups, 100 iterations, five alternating-order
  repeats; median
- Timed pipeline: paired gate/up, fused SwiGLU/Hadamard and accepted M64
  down/scatter

## Results

Every tested row count produced bit-identical gate/up, post-SwiGLU/Hadamard and
down/scatter output tensors (`max_abs=0`). Delta is candidate versus accepted
M64; positive is slower.

| Expert rows | Accepted M64 (ms) | Parallel pair (ms) | Delta |
| ---: | ---: | ---: | ---: |
| 129 | 0.1133 | 0.1785 | +57.6% |
| 145 | 0.1179 | 0.1788 | +51.6% |
| 192 | 0.1205 | 0.1820 | +51.0% |
| 255 | 0.1661 | 0.2024 | +21.9% |
| 257 | 0.1707 | 0.2041 | +19.6% |
| 320 | 0.1877 | 0.2260 | +20.4% |
| 383 | 0.2045 | 0.2289 | +11.9% |
| 385 | 0.2413 | 0.2901 | +20.2% |
| 512 | 0.2805 | 0.3248 | +15.8% |
| 640 | 0.3526 | 0.3615 | +2.5% |
| 768 | 0.3960 | 0.4009 | +1.3% |
| 1,024 | 0.5463 | 0.5756 | +5.4% |
| 2,048 | 1.1594 | 1.2406 | +7.0% |
| 4,096 | 2.3510 | 2.4725 | +5.2% |
| 6,528 | 3.8011 | 3.9513 | +4.0% |

## Consequence

The result rules out both obvious forms of gate/up A-tile co-residency:

1. eight warps computing both matrices serially loses from doubled live state
   and reduced parallelism;
2. sixteen warps computing the matrices independently retains 64 registers but
   still loses from the 512-thread CTA and doubled shared pipeline/scratch.

The accepted independent M64 CTA is the better GB10 scheduling unit. The next
expert-kernel experiments should target K4 codebook decode instructions,
Hadamard/output traffic, or the stock thin-expert kernel without enlarging or
co-locating CTAs.
