// SPDX-License-Identifier: AGPL-3.0-only
// Ordered bf16 GEMM C = A * B^T, M=3, N=288, K=4096, row-major, sm_121a.
//
// Arithmetic contract (per output (m,n), mandatory exact order):
//   float acc = +0.0f;
//   for k = 0..4095: acc = __fadd_rn(acc, __fmul_rn(bf2f(A[m][k]), bf2f(B[n][k])));
//   C[m][n] = __float2bfloat16_rn(acc);
// One serial FP32 dependency chain per output; no split-K, no partial
// accumulators, no FMA contraction (explicit __fmul_rn/__fadd_rn, compiled
// with --fmad=false), no reassociation, no tensor cores.
//
// Structural improvement over a 16x16-tile / K-chunk-staged reference:
// 1. Output mapping is 3 x 85 per block (255 of 256 threads compute; the
//    reference wastes 13 of 16 row lanes because M=3 < 16). All 288 columns
//    are covered by 4 blocks; only lane/col bound checks mask the tail.
// 2. The entire A matrix (3*4096 bf16 = exactly 24 KiB) is staged in shared
//    memory ONCE per block, cooperatively, with a single __syncthreads()
//    after the load. Thereafter no barriers are needed at all: each thread
//    owns one output and walks k = 0..4095 sequentially, reading its A row
//    from shared and its B row (n fixed, read by the 3 same-lane threads of
//    the block -> hot in L1) directly from global memory. This replaces the
//    reference's 2 barriers x 256 chunks = 512 barriers with 1 barrier.
// 3. A is loaded with uint4 vectors when A is 16-byte aligned: A holds
//    24576 B = 1536 uint4 vectors, loaded via a bounded grid-stride loop
//    (6 vectors per thread at stride 256, threads 0..255; 255*6 < 1536 <
//    256*6, so vector index bounds are exact, no tail access outside A).
//    A misaligned A falls back to a scalar shared-memory load; both paths
//    touch only bytes inside A. The shared staging buffer is declared with
//    alignas(16) so the uint4 stores are validly aligned.
//
// Bounds/synchronization proof: every thread (0..255) participates in the
// shared load and the one __syncthreads() before any return, so the barrier
// is uniformly reached. Vector path: indices i < 1536 with i = tid + 256*j,
// j < 6, exactly cover A's 1536 uint4 vectors. Scalar path:
// i = tid + 256*j < 12288 for j < 48, exactly covering A. After the
// barrier, sA[0..12287] is fully written. A thread with m>=3 or n>=288
// exits after the barrier and never reads or writes C; each valid (m,n) is
// produced by exactly one thread, so every C element is written exactly
// once. B and A are never written.
//
// Launch geometry: grid = 4 blocks (blockIdx.x*85 + 85 >= 288 for bx=3:
// columns 255..287 valid), block = 256 threads, static shared memory of
// exactly 24 KiB (default limit, no opt-in required).

#ifndef ORDERED_BF16_3X288_CUH
#define ORDERED_BF16_3X288_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace obm {

constexpr int K = 4096;
constexpr int M = 3;
constexpr int N = 288;
constexpr int COLS_PER_BLOCK = 85;      // 3*85 = 255 <= 256 threads
constexpr int BLOCK = 256;
constexpr int GRID = (N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK; // 4
constexpr int A_VECS = M * K * 2 / 16;  // 1536 uint4 vectors cover all of A

template <bool ALIGNED>
__global__ void __launch_bounds__(BLOCK) gemm3x288_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C) {
  alignas(16) __shared__ __nv_bfloat16 sA[M * K];   // exactly 24 KiB

  // One-time cooperative staging of all of A into shared memory.
  if (ALIGNED) {
    // Grid-stride uint4 load: 1536 vectors / 256 threads = exactly 6 each.
    const uint4* src = reinterpret_cast<const uint4*>(A);
    uint4* dst = reinterpret_cast<uint4*>(sA);
    for (int i = threadIdx.x; i < A_VECS; i += BLOCK)
      dst[i] = src[i];
  } else {
    for (int i = threadIdx.x; i < M * K; i += BLOCK)  // i < 12288 always
      sA[i] = A[i];
  }
  __syncthreads();                      // single barrier: sA fully visible

  const int t    = threadIdx.x;
  const int m    = t / COLS_PER_BLOCK;                 // 0..2 for t<255
  const int lane = t % COLS_PER_BLOCK;
  const int n    = blockIdx.x * COLS_PER_BLOCK + lane; // may be >= N
  if (m >= M || n >= N) return;       // tail mask; A staging already done

  const __nv_bfloat16* Arow = sA + (size_t)m * K;
  const __nv_bfloat16* Brow = B + (size_t)n * K;

  // The single mandated FP32 dependency chain, k strictly in order.
  float acc = +0.0f;
#pragma unroll 4
  for (int k = 0; k < K; ++k) {
    float product = __fmul_rn(__bfloat162float(Arow[k]),
                              __bfloat162float(Brow[k]));
    acc = __fadd_rn(acc, product);
  }
  C[(size_t)m * N + n] = __float2bfloat16_rn(acc);
}

} // namespace obm

extern "C" cudaError_t launch_ordered_bf16_3x288(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    cudaStream_t stream) {
  if (!A || !B || !C) return cudaErrorInvalidValue;
  const bool aligned = (reinterpret_cast<uintptr_t>(A) & 0xF) == 0;
  if (aligned)
    obm::gemm3x288_kernel<true><<<obm::GRID, obm::BLOCK, 0, stream>>>(A, B, C);
  else
    obm::gemm3x288_kernel<false><<<obm::GRID, obm::BLOCK, 0, stream>>>(A, B, C);
  return cudaGetLastError();
}

#endif // ORDERED_BF16_3X288_CUH
