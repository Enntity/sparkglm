# EXL3 M64 Nsight Compute profile

Date: 2026-09-03

## Verdict

The accepted M64 fat-expert kernels are latency and synchronization limited,
not close to either GB10 tensor or memory throughput. The next justified
experiment is the shared-memory epilogue layout: Nsight Compute attributes
roughly 61% of shared-store wavefronts to bank conflicts and substantial warp
delay to barriers around that scratch exchange.

This profile also explains why simply adding more work to a CTA lost. The
kernel needs more eligible warps and a cheaper cross-warp Hadamard exchange,
not a larger scheduling unit.

## Method

- Image: `sparkglm-vllm:exl3-m32-candidate-20260903a`
- Accepted kernels profiled: unchanged M64 paired gate/up and M64 down/scatter
- Target: one GB10, `sm_121a`
- Shape: 512 expert rows, hidden 4,096, TP-local intermediate 1,024
- Tool: Nsight Compute 2025.3.1, full set, 38/39 replay passes
- Both M64 outputs remained bit-identical to the M128 oracle

The image contains additive M32 symbols, but the profiled M64 source and launch
geometry are identical to the accepted `9029cad8e8` implementation.

## Paired gate/up M64

| Counter | Value |
| --- | ---: |
| Grid / block | 128 CTAs / 256 threads |
| Waves per SM | 0.67 |
| Registers per thread | 64 |
| Dynamic shared memory | 17.41 KiB |
| Compute throughput | 30.72% |
| Memory throughput | 29.08% |
| Achieved occupancy | 42.32% |
| Theoretical occupancy | 66.67% |
| Eligible warps per scheduler | 0.52 |
| No-eligible-warp cycles | 73.37% |
| Shared-store bank conflicts | 102,110 |
| Conflicted share of shared-store wavefronts | 61.00% |
| Excess shared wavefronts | 23% |

Nsight's leading sampled stall was waiting for sibling warps at a CTA barrier:
7.9 of 19.2 average cycles between issued instructions, or 40.88%.

## Down/scatter M64

| Counter | Value |
| --- | ---: |
| Grid / block | 256 CTAs / 256 threads |
| Waves per SM | 1.33 |
| Registers per thread | 64 |
| Dynamic shared memory | 17.41 KiB |
| Compute throughput | 20.32% |
| Memory throughput | 20.32% |
| Achieved occupancy | 53.90% |
| Theoretical occupancy | 66.67% |
| Eligible warps per scheduler | 0.48 |
| No-eligible-warp cycles | 77.16% |
| Shared-store bank conflicts | 202,659 |
| Conflicted share of shared-store wavefronts | 60.91% |
| Excess shared wavefronts | 25% |

The leading sampled down/scatter stall was an L1TEX scoreboard dependency:
12.2 of 28.6 average cycles between issued instructions, or 42.63%. The launch
also has one full wave plus a 64-CTA partial wave.

## Source mechanism and next gate

Each warp owns one 16-column output block, but fragment values are staged into
a shared `[16][128]` row-major scratch before a warp performs the required
128-wide Hadamard. A 128-float row stride maps equivalent columns of all rows
to the same bank. The fragment-store lane pattern therefore creates heavy bank
conflicts before the block-wide barrier.

Test padded shared row strides while retaining the exact M64/K16/N128 compute
and independent 256-thread CTAs. A candidate must be bit-exact and win the full
paired gate/up + activation + down/scatter pipeline across the production row
distribution before an endpoint build is justified.
