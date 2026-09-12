// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Pinned FlashKDA normalizes Q/K internally with epsilon=1e-6.
// Preserve raw BF16 operands; a second normalization corrupts tiny Q/K.
extern "C" __global__ void atlas_flash_pack(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ raw_beta,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ beta_ht,
    unsigned int tokens, unsigned int heads) {
    const unsigned int head = blockIdx.x, t = blockIdx.y, row = threadIdx.x;
    if (head >= heads || t >= tokens || row >= 128) return;
    const unsigned long long base = (unsigned long long)t * 3 * heads * 128 + head * 128 + row;
    const unsigned long long off = ((unsigned long long)t * heads + head) * 128 + row;
    q[off] = qkv[base];
    k[off] = qkv[base + (unsigned long long)heads * 128];
    v[off] = qkv[base + (unsigned long long)heads * 256];
    if (row == 0) beta_ht[(unsigned long long)head * tokens + t] = raw_beta[(unsigned long long)t * heads + head];
}

extern "C" __global__ void atlas_state_transpose(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int heads
) {
    __shared__ float tile[32][33];
    const unsigned int x = blockIdx.x * 32 + threadIdx.x;
    const unsigned int y = blockIdx.y * 32 + threadIdx.y;
    const unsigned long long base = (unsigned long long)blockIdx.z * 128 * 128;
    for (unsigned int j = 0; j < 32; j += 8) {
        tile[threadIdx.y + j][threadIdx.x] = src[base + (unsigned long long)(y + j) * 128 + x];
    }
    __syncthreads();
    const unsigned int rx = blockIdx.y * 32 + threadIdx.x;
    const unsigned int ry = blockIdx.x * 32 + threadIdx.y;
    for (unsigned int j = 0; j < 32; j += 8) {
        dst[base + (unsigned long long)(ry + j) * 128 + rx] = tile[threadIdx.x][threadIdx.y + j];
    }
}
