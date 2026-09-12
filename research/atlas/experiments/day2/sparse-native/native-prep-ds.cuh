// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

// q[rows,32,512] -> out[rows,32,576]; copy first 512, zero last 64
__global__ void prep_q(const __nv_bfloat16* __restrict__ q,
                       __nv_bfloat16* __restrict__ out,
                       unsigned rows) {
    const size_t total = (size_t)rows * 32u * 576u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t lane = i % 576u;
        if (lane < 512u) {
            const size_t src = (i / 576u) * 512u + lane;
            out[i] = q[src];
        } else {
            out[i] = __nv_bfloat16(0.0f);
        }
    }
}

// paged latent[pblocks,16,512], logical token t -> blocktable[t/16]*16 + t%16
// one CTA (128 threads / 4 warps) per logical or padded token
__global__ void prep_kv(const __nv_bfloat16* __restrict__ paged,
                        const int32_t* __restrict__ blocktable,
                        unsigned char* __restrict__ out,
                        unsigned seqend) {
    __shared__ float smem[128];
    __shared__ float sscale[4];

    const unsigned t = blockIdx.x;
    const int tid = threadIdx.x;

    if (t >= seqend) {
        // padded token: first 512 zero, four fp32 scales = 1, final 128 zero
        for (int i = tid; i < 512; i += 128) out[(size_t)t * 656u + i] = 0;
        if (tid < 4) ((float*)(out + (size_t)t * 656u + 512u))[tid] = 1.0f;
        for (int i = tid; i < 128; i += 128) out[(size_t)t * 656u + 528u + i] = 0;
        return;
    }

    const int32_t blk = blocktable[t >> 4];
    const size_t row = ((size_t)blk << 4) + (size_t)(t & 15u);
    const __nv_bfloat16* src = paged + row * 512u;

    const float x = __bfloat162float(src[0 * 128 + tid]); // replace below
    (void)x;

    for (int g = 0; g < 4; ++g) {
        const float v = __bfloat162float(src[g * 128 + tid]);
        const float a = fabsf(v);
        smem[tid] = a;
        __syncthreads();
        // tree reduction of 128 values
        for (int s = 64; s > 0; s >>= 1) {
            if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
            __syncthreads();
        }
        if (tid == 0) {
            float m = smem[0];
            if (!(m > 1.0e-4f)) m = 1.0e-4f;
            sscale[g] = m / 448.0f;
        }
        __syncthreads(); // sscale[g] visible; also protects smem reuse
        const float sc = sscale[g];
        out[(size_t)t * 656u + (size_t)(g * 128 + tid)] =
            (unsigned char)__nv_cvt_float_to_fp8(v / sc, __NV_SATFINITE, __NV_E4M3);
    }

    if (tid < 4)
        ((float*)(out + (size_t)t * 656u + 512u))[tid] = sscale[tid];
    for (int i = tid; i < 128; i += 128)
        out[(size_t)t * 656u + 528u + i] = 0;
}
