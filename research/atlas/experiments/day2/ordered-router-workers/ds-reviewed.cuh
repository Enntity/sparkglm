// SPDX-License-Identifier: AGPL-3.0-only
// C[m,n] = sum_k A[m,k]*B[n,k], M=3, N=288, K=4096, ordered FP32 chain:
// acc=+0.0f; for k in 0..4095: acc=__fadd_rn(acc, __fmul_rn(a,b)); store RN(BF16).
// Exactly one accumulator per output, k ascending, no FMA/reassociation.
//
// Geometry: M=3 is tiny. Block handles 32 output columns (n) and ALL 3 rows,
// so every row is used. We tile n into blocks of 32 -> 9 blocks along N
// (288/32). Grid = (9). Block = 96 threads (3 rows x 32 cols).
// Each thread owns exactly one output element (m,n) and runs the full K loop
// in order. A shared staging tile of B for the 32 columns is loaded cooperatively
// by 96 threads with float4-sized 16-byte vector loads of BF16 (8 elements/load),
// then two __syncthreads() per chunk guarantee the tile is complete/consumed.
// No split-K, no multi-accumulator, no tree reduce. Bounds: K=4096 divisible by
// chunk 64 -> vector loads land exactly; N=288 divisible by 32 -> no tail cols.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define CHUNK 64            // K elements staged per shared-memory pass (even, /8)
#define BLKCOLS 32          // output columns per block (n)
#define NTHREADS (3*BLKCOLS) // 96 threads

__global__ void __launch_bounds__(NTHREADS, 1)
ordered_bf16_kernel(const __nv_bfloat16* __restrict__ A,
                    const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C) {
    // Shared tile for B: BLKCOLS columns x CHUNK elements. 32*64*2B = 4KiB.
    __shared__ __align__(16) __nv_bfloat16 sB[BLKCOLS][CHUNK];
    // A tile: 3 rows x CHUNK elements (only 3 valid rows, all threads may read).
    __shared__ __align__(16) __nv_bfloat16 sA[3][CHUNK];

    const int n0 = blockIdx.x * BLKCOLS;         // first output column of block
    const int tid = threadIdx.x;                 // 0..95
    // Thread mapping: tid = m*BLKCOLS + c  -> m in 0..2, c in 0..31 (local col).
    const int m = tid / BLKCOLS;                 // 0..2 (all valid: M=3 rows)
    const int c = tid % BLKCOLS;                 // local output column
    const int n = n0 + c;                        // global output column (valid)

    float acc = 0.0f;                            // FP32 accumulator (BF16->f32)

    for (int k0 = 0; k0 < 4096; k0 += CHUNK) {
        // ---- cooperative 16B vector loads: CHUNK=64 elements = 8 loads/row ----
        // sA: 3 rows x 64 elems -> 192 elems -> 24 float4 loads over 96 threads.
        // sB: 32 rows x 64 elems -> 2048 elems -> 256 float4 loads over 96 threads.
        // Each float4 covers 8 BF16 (16 bytes). Loads never cross matrix bounds:
        // k0+7 <= 4095 since k0 multiple of 64 and 4096 divisible by 64.
        for (int idx = tid; idx < 3 * (CHUNK / 8); idx += NTHREADS) {
            const int r = idx / (CHUNK / 8);
            const int off = idx % (CHUNK / 8);
            const float4 v = *reinterpret_cast<const float4*>(
                &A[r * 4096 + k0 + off * 8]);
            *reinterpret_cast<float4*>(&sA[r][off * 8]) = v;
        }
        for (int idx = tid; idx < BLKCOLS * (CHUNK / 8); idx += NTHREADS) {
            const int r = idx / (CHUNK / 8);
            const int off = idx % (CHUNK / 8);
            const float4 v = *reinterpret_cast<const float4*>(
                &B[(n0 + r) * 4096 + k0 + off * 8]);
            *reinterpret_cast<float4*>(&sB[r][off * 8]) = v;
        }
        __syncthreads();                         // tile complete for all threads

        // ---- ordered accumulation: k ascending within chunk, thread-local ----
        const __nv_bfloat16* arow = &sA[m][0];
        const __nv_bfloat16* brow = &sB[c][0];
#pragma unroll
        for (int k = 0; k < CHUNK; ++k) {
            const float a = __bfloat162float(arow[k]);
            const float b = __bfloat162float(brow[k]);
            const float p = __fmul_rn(a, b);     // rounded product
            acc = __fadd_rn(acc, p);             // rounded add -> exact chain
        }
        __syncthreads();                         // safe to overwrite tile next pass

        // Compiler cannot reorder across the chain (fmad=false, strict seq).
        // k0 increments by 64, so global k order 0,1,...,4095 is preserved.
    }

    C[m * 288 + n] = __float2bfloat16_rn(acc);   // single rounded BF16 store
}

extern "C" cudaError_t launch_ordered_bf16_3x288(const __nv_bfloat16* A,
                                                 const __nv_bfloat16* B,
                                                 __nv_bfloat16* C,
                                                 cudaStream_t stream) {
    if (A == nullptr || B == nullptr || C == nullptr) return cudaErrorInvalidValue;
    dim3 grid(288 / BLKCOLS);   // 9 blocks
    dim3 block(NTHREADS);       // 96 threads
    ordered_bf16_kernel<<<grid, block, 0, stream>>>(A, B, C);
    return cudaGetLastError();
}
