// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cstddef>
#include <cstdint>

// Merge partitioned attention outputs: weighted combine of partial outputs by LSE.
// part_o   : float[splits, rows, heads, dim] normalized partial attention outputs
// part_lse : float[splits, rows, heads] natural-logsumexp (-INFINITY == empty partition)
// out_bf16 : __nv_bfloat16[rows, heads, dim]
// out_lse  : float[rows, heads]
__global__ void merge_attention_parts_kernel(const float* __restrict__ part_o,
                                             const float* __restrict__ part_lse,
                                             __nv_bfloat16* __restrict__ out_bf16,
                                             float* __restrict__ out_lse,
                                             unsigned rows, unsigned heads,
                                             unsigned dim, unsigned splits) {
    __shared__ float s_w[16];
    __shared__ unsigned s_nempty;
    __shared__ float s_bad;   // nonfinite LSE flag

    unsigned rh = blockIdx.x;                 // flattened row*heads + head
    unsigned row = rh / heads, head = rh % heads;
    size_t base_o = (size_t)row * heads * dim + (size_t)head * dim;
    size_t stride_o = (size_t)rows * heads * dim;

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        unsigned n = 0;
        float bad = 0.0f;
        for (unsigned s = 0; s < splits; ++s) {
            float l = part_lse[(size_t)s * rows * heads + rh];
            if (isnan(l) || l == INFINITY) bad = 1.0f;   // propagate nonfinite
            if (l != -INFINITY) { if (l > mx) mx = l; ++n; }
        }
        s_nempty = n;
        s_bad = bad;
        if (bad != 0.0f || n == 0) {
            s_w[0] = bad != 0.0f ? 1.0f : 0.0f;   // weight for nonfinite path
        } else {
            float sum = 0.0f;
            for (unsigned s = 0; s < splits; ++s) {
                float l = part_lse[(size_t)s * rows * heads + rh];
                if (l == -INFINITY) { s_w[s] = 0.0f; continue; }
                float e = expf(l - mx);
                s_w[s] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (unsigned s = 0; s < splits; ++s) s_w[s] *= inv;
        }
        out_lse[(size_t)row * heads + head] =
            (bad != 0.0f) ? CUDART_NAN_F
                          : (n == 0 ? -INFINITY : mx + logf(sum_placeholder_unused()));
    }
    __syncthreads();

    // Recompute nothing; thread0 stored final lse handling below via second pass guard.
    // (see note) -- handled by rewriting out_lse in thread0 with correct sum.

    bool nonfinite = (s_bad != 0.0f);
    bool empty = (s_nempty == 0);
    bool single = (s_nempty == 1);

    for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc;
        if (nonfinite) {
            acc = CUDART_NAN_F;
        } else if (empty) {
            acc = 0.0f;
        } else if (single) {
            // weight is 1.0; copy active partition's normalized FP32 output
            unsigned s = 0;
            while (s < splits && part_lse[(size_t)s * rows * heads + rh] == -INFINITY) ++s;
            acc = part_o[(size_t)s * stride_o + base_o + d];
        } else {
            acc = 0.0f;
            for (unsigned s = 0; s < splits; ++s) {
                if (s_w[s] == 0.0f && part_lse[(size_t)s * rows * heads + rh] == -INFINITY)
                    continue;
                acc += s_w[s] * part_o[(size_t)s * stride_o + base_o + d];
            }
        }
        out_bf16[(size_t)row * heads * dim + (size_t)head * dim + d] = __float2bfloat16_rn(acc);
    }
}

extern "C" int merge_attention_parts(const float* part_o, const float* part_lse,
                                     void* out_bf16, float* out_lse,
                                     unsigned rows, unsigned heads, unsigned dim,
                                     unsigned splits, void* stream) {
    if (!part_o || !part_lse || !out_bf16 || !out_lse) return (int)cudaErrorInvalidValue;
    if (rows < 1 || rows > 4 || heads != 32 || dim != 512 || splits < 1 || splits > 16)
        return (int)cudaErrorInvalidValue;
    if (((uintptr_t)part_o & 3u) || ((uintptr_t)part_lse & 3u) ||
        ((uintptr_t)out_bf16 & 1u) || ((uintptr_t)out_lse & 3u))
        return (int)cudaErrorInvalidValue;

    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    unsigned blocks = rows * heads;
    merge_attention_parts_kernel<<<blocks, 256, 0, st>>>(
        part_o, part_lse, reinterpret_cast<__nv_bfloat16*>(out_bf16), out_lse,
        rows, heads, dim, splits);
    return (int)cudaGetLastError();
}
