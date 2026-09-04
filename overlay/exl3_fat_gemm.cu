// SPDX-License-Identifier: MIT AND Apache-2.0
//
// The direct K4 trellis decode, Hadamard helpers, and tensor-core MMA layout
// are derived from ExLlamaV3.  The fat-GEMM cp.async pipeline was adapted from
// Reederey87/glm53-flash-exl3-2x-dgx-spark@0c03250.  The GPU-resident grouped
// task planner and five-phase GLM-5.3 TP2 prefill execution contract are
// original to this repository; see docs/PROVENANCE.md.

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"
#include "exl3_fat_gemm.cuh"

namespace {

constexpr int FAT_THREADS = 256;
constexpr int FAT_TILE_M = 128;
constexpr int FAT_TILE_M_SMALL = 64;
constexpr int FAT_TILE_K = 16;
constexpr int FAT_TILE_N = 128;
constexpr int FAT_N_BLOCKS = FAT_TILE_N / 16;
constexpr int FAT_PACKED_WORDS = 4 * 16;
constexpr float HAD_SCALE = 0.088388347648f;

extern "C" __device__ float __nv_expf(float);

__device__ inline float rounded_swiglu(float gate, float up)
{
    float denominator = __fadd_rn(1.0f, __nv_expf(-gate));
    float sigmoid = __fdiv_rn(1.0f, denominator);
    float silu = __fmul_rn(sigmoid, gate);
    return __fmul_rn(silu, up);
}

__global__ __launch_bounds__(32)
void exl3_fat_swiglu_had_kernel(
    const float* __restrict__ gate_up,
    half* __restrict__ out,
    const half* __restrict__ down_suh,
    int intermediate,
    float limit)
{
    int lane = threadIdx.x;
    int row = blockIdx.x;
    int col = blockIdx.y * 128 + lane * 4;
    int input_row = row * 2 * intermediate;
    float4 gate = reinterpret_cast<const float4*>(gate_up + input_row + col)[0];
    float4 up = reinterpret_cast<const float4*>(
        gate_up + input_row + intermediate + col)[0];

    gate.x = fminf(gate.x, limit);
    gate.y = fminf(gate.y, limit);
    gate.z = fminf(gate.z, limit);
    gate.w = fminf(gate.w, limit);
    up.x = fminf(fmaxf(up.x, -limit), limit);
    up.y = fminf(fmaxf(up.y, -limit), limit);
    up.z = fminf(fmaxf(up.z, -limit), limit);
    up.w = fminf(fmaxf(up.w, -limit), limit);

    gate.x = rounded_swiglu(gate.x, up.x);
    gate.y = rounded_swiglu(gate.y, up.y);
    gate.z = rounded_swiglu(gate.z, up.z);
    gate.w = rounded_swiglu(gate.w, up.w);

    half4 value;
    value.x = __floats2half2_rn(gate.x, gate.y);
    value.y = __floats2half2_rn(gate.z, gate.w);
    half4 scale = reinterpret_cast<const half4*>(down_suh + col)[0];
    value.x = __hmul2(value.x, scale.x);
    value.y = __hmul2(value.y, scale.y);

    float v0 = __half2float(__low2half(value.x));
    float v1 = __half2float(__high2half(value.x));
    float v2 = __half2float(__low2half(value.y));
    float v3 = __half2float(__high2half(value.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    value.x = __floats2half2_rn(h0 * HAD_SCALE, h1 * HAD_SCALE);
    value.y = __floats2half2_rn(h2 * HAD_SCALE, h3 * HAD_SCALE);
    reinterpret_cast<half4*>(out + row * intermediate + col)[0] = value;
}

__device__ inline void fat_had_ff_128(
    const float* input_ptr,
    float* output_ptr,
    const half* scale)
{
    int lane = threadIdx.x & 31;
    float4 v = reinterpret_cast<const float4*>(input_ptr)[lane];

    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    v.x *= HAD_SCALE;
    v.y *= HAD_SCALE;
    v.z *= HAD_SCALE;
    v.w *= HAD_SCALE;

    half4 scales = reinterpret_cast<const half4*>(scale)[lane];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
    reinterpret_cast<float4*>(output_ptr)[lane] = v;
}

template <int tile_m, bool scatter, bool paired>
__global__ __launch_bounds__(FAT_THREADS)
void exl3_fat_gemm_kernel(
    const half* __restrict__ a,
    const uint16_t* __restrict__ packed,
    const uint16_t* __restrict__ packed_pair,
    float* __restrict__ out,
    const half* __restrict__ svh,
    const half* __restrict__ svh_pair,
    const int64_t* __restrict__ token_idx,
    const half* __restrict__ route_weight,
    int size_m,
    int size_k,
    int size_n,
    int out_stride_n)
{
    static_assert(tile_m == FAT_TILE_M || tile_m == FAT_TILE_M_SMALL);
    constexpr int tile_m_blocks = tile_m / 16;
    extern __shared__ unsigned char shared_raw[];
    // S2b: 3-stage cp.async pipeline. Stages cycle so that issue(j+2) targets
    // the buffer compute(j) is done with, and sync(j+1) certifies every warp
    // finished compute(j) before that issue fires — the single per-iteration
    // barrier below is the only one the k-loop needs.
    constexpr int FAT_STAGES = 3;
    half* sh_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(
        sh_a + FAT_STAGES * tile_m * FAT_TILE_K);
    float* sh_c = reinterpret_cast<float*>(
        sh_b + FAT_STAGES * FAT_N_BLOCKS * FAT_PACKED_WORDS);

    int t = threadIdx.x;
    int warp = t / 32;
    int lane = t & 31;
    int m_base = blockIdx.y * tile_m;
    int n_base = blockIdx.x * FAT_TILE_N;
    int out_n_base = n_base;
    if constexpr (paired)
    {
        if (n_base >= size_n)
        {
            n_base -= size_n;
            out_n_base = size_n + n_base;
            packed = packed_pair;
            svh = svh_pair;
        }
    }
    int tiles_n = size_n / 16;

    FragC frag_c[tile_m_blocks][2];
    #pragma unroll
    for (int mb = 0; mb < tile_m_blocks; ++mb)
    {
        frag_c[mb][0] = {};
        frag_c[mb][1] = {};
    }

    // Issue one k_block's global->SMEM tile into `stage` (async, 16B cp.async).
    // A: 256 threads, one int4 each, XOR-swizzled destination; OOB rows are
    // zero-filled synchronously (keeps the stock kernel's exact semantics;
    // upstream notes cp_async_pred miscompiles on Blackwell — use if-form).
    // B: first 64 threads, one int4 each.
    auto issue_stage = [&](int k_block, int stage)
    {
        if (t < 2 * tile_m)
        {
            int a_row = t / 2;
            int a_col8 = t & 1;
            int a_dst_col8 = a_col8 ^ ((a_row >> 2) & 1);
            int4* a_dst = reinterpret_cast<int4*>(
                sh_a + stage * tile_m * FAT_TILE_K) + a_row * 2 + a_dst_col8;
            if (m_base + a_row < size_m)
            {
                const int4* a_src = reinterpret_cast<const int4*>(
                    a + (m_base + a_row) * size_k + k_block * FAT_TILE_K);
                cp_async(a_dst, a_src + a_col8);
            }
            else
            {
                *a_dst = int4{};
            }
        }
        if (t < 64)
        {
            const int4* b_src = reinterpret_cast<const int4*>(
                packed + (k_block * tiles_n + n_base / 16) * FAT_PACKED_WORDS);
            cp_async(
                reinterpret_cast<int4*>(
                    sh_b + stage * FAT_N_BLOCKS * FAT_PACKED_WORDS) + t,
                b_src + t);
        }
    };

    int const k_blocks = size_k / FAT_TILE_K;
    if (k_blocks > 0)
    {
        issue_stage(0, 0);
        cp_async_fence();
    }

    for (int k_block = 0; k_block < k_blocks; ++k_block)
    {
        if (k_block + 1 < k_blocks)
            issue_stage(k_block + 1, (k_block + 1) % FAT_STAGES);
        cp_async_fence();
        cp_async_wait<FAT_STAGES - 2>();  // stage k_block's group is complete
        __syncthreads();

        int const stage = k_block % FAT_STAGES;
        FragB frag_b0;
        FragB frag_b1;
        const uint32_t* warp_b = reinterpret_cast<const uint32_t*>(
            sh_b + stage * FAT_N_BLOCKS * FAT_PACKED_WORDS + warp * FAT_PACKED_WORDS);
        dq_dispatch<4, 1>(warp_b, lane << 3, frag_b0, frag_b1);

        #pragma unroll
        for (int mb = 0; mb < tile_m_blocks; ++mb)
        {
            FragA frag_a;
            int row = (lane % 8) + 8 * ((lane / 8) % 2) + mb * 16;
            int base_col = lane / 16;
            int swizzled_col = base_col ^ ((row >> 2) & 1);
            ldsm4(frag_a, reinterpret_cast<int4*>(
                sh_a + stage * tile_m * FAT_TILE_K) + row * 2 + swizzled_col);
            ptx_mma_m16n8k16(frag_a, frag_b0, frag_c[mb][0]);
            ptx_mma_m16n8k16(frag_a, frag_b1, frag_c[mb][1]);
        }
    }

    #pragma unroll
    for (int mb = 0; mb < tile_m_blocks; ++mb)
    {
        int rows = min(16, size_m - (m_base + mb * 16));
        if (rows <= 0) break;
        int row0 = lane / 4;
        int row1 = row0 + 8;
        int col = (lane % 4) * 2;
        int n0 = warp * 16;
        if (row0 < rows)
        {
            float* dst0 = sh_c + row0 * FAT_TILE_N + n0 + col;
            dst0[0] = frag_c[mb][0][0];
            dst0[1] = frag_c[mb][0][1];
            dst0[8] = frag_c[mb][1][0];
            dst0[9] = frag_c[mb][1][1];
        }
        if (row1 < rows)
        {
            float* dst1 = sh_c + row1 * FAT_TILE_N + n0 + col;
            dst1[0] = frag_c[mb][0][2];
            dst1[1] = frag_c[mb][0][3];
            dst1[8] = frag_c[mb][1][2];
            dst1[9] = frag_c[mb][1][3];
        }
        __syncthreads();

        for (int row = warp; row < rows; row += 8)
        {
            fat_had_ff_128(
                sh_c + row * FAT_TILE_N,
                sh_c + row * FAT_TILE_N,
                svh + n_base);
        }
        __syncthreads();

        for (int i = t; i < rows * FAT_TILE_N; i += FAT_THREADS)
        {
            int row = i / FAT_TILE_N;
            int col_out = i % FAT_TILE_N;
            int source_row = m_base + mb * 16 + row;
            float value = sh_c[i];
            if constexpr (scatter)
            {
                int64_t destination = token_idx[source_row];
                value *= __half2float(route_weight[source_row]);
                // One route per token reaches a given expert, and expert
                // launches share this stream, so this accumulation is race-free.
                out[destination * out_stride_n + out_n_base + col_out] += value;
            }
            else
            {
                out[source_row * out_stride_n + out_n_base + col_out] = value;
            }
        }
        __syncthreads();
    }
}

// One M64xN128 direct-trellis tile used by the grouped prefill kernel below.
// The arithmetic and cp.async pipeline intentionally mirror
// exl3_fat_gemm_kernel<64>; the new work is the GPU-resident orchestration,
// not a different quantization result.
template <bool scatter, bool atomic_scatter = false>
__device__ __forceinline__ void grouped_fat_m64_tile(
    const half* __restrict__ a,
    const uint16_t* __restrict__ packed,
    float* __restrict__ out,
    const half* __restrict__ svh,
    const int64_t* __restrict__ token_idx,
    const half* __restrict__ route_weight,
    int size_m,
    int size_k,
    int size_n,
    int out_stride_n,
    int m_base,
    int n_base,
    int out_n_base,
    half* sh_a,
    uint16_t* sh_b,
    float* sh_c)
{
    constexpr int tile_m = FAT_TILE_M_SMALL;
    constexpr int tile_m_blocks = tile_m / 16;
    constexpr int FAT_STAGES = 3;
    int t = threadIdx.x;
    int warp = t / 32;
    int lane = t & 31;
    int tiles_n = size_n / 16;

    FragC frag_c[tile_m_blocks][2];
    #pragma unroll
    for (int mb = 0; mb < tile_m_blocks; ++mb)
    {
        frag_c[mb][0] = {};
        frag_c[mb][1] = {};
    }

    auto issue_stage = [&](int k_block, int stage)
    {
        if (t < 2 * tile_m)
        {
            int a_row = t / 2;
            int a_col8 = t & 1;
            int a_dst_col8 = a_col8 ^ ((a_row >> 2) & 1);
            int4* a_dst = reinterpret_cast<int4*>(
                sh_a + stage * tile_m * FAT_TILE_K) + a_row * 2 + a_dst_col8;
            if (m_base + a_row < size_m)
            {
                const int4* a_src = reinterpret_cast<const int4*>(
                    a + (m_base + a_row) * size_k + k_block * FAT_TILE_K);
                cp_async(a_dst, a_src + a_col8);
            }
            else
            {
                *a_dst = int4{};
            }
        }
        if (t < 64)
        {
            const int4* b_src = reinterpret_cast<const int4*>(
                packed + (k_block * tiles_n + n_base / 16) * FAT_PACKED_WORDS);
            cp_async(
                reinterpret_cast<int4*>(
                    sh_b + stage * FAT_N_BLOCKS * FAT_PACKED_WORDS) + t,
                b_src + t);
        }
    };

    int const k_blocks = size_k / FAT_TILE_K;
    issue_stage(0, 0);
    cp_async_fence();
    for (int k_block = 0; k_block < k_blocks; ++k_block)
    {
        if (k_block + 1 < k_blocks)
            issue_stage(k_block + 1, (k_block + 1) % FAT_STAGES);
        cp_async_fence();
        cp_async_wait<FAT_STAGES - 2>();
        __syncthreads();

        int const stage = k_block % FAT_STAGES;
        FragB frag_b0;
        FragB frag_b1;
        const uint32_t* warp_b = reinterpret_cast<const uint32_t*>(
            sh_b + stage * FAT_N_BLOCKS * FAT_PACKED_WORDS
            + warp * FAT_PACKED_WORDS);
        dq_dispatch<4, 1>(warp_b, lane << 3, frag_b0, frag_b1);

        #pragma unroll
        for (int mb = 0; mb < tile_m_blocks; ++mb)
        {
            FragA frag_a;
            int row = (lane % 8) + 8 * ((lane / 8) % 2) + mb * 16;
            int base_col = lane / 16;
            int swizzled_col = base_col ^ ((row >> 2) & 1);
            ldsm4(frag_a, reinterpret_cast<int4*>(
                sh_a + stage * tile_m * FAT_TILE_K) + row * 2 + swizzled_col);
            ptx_mma_m16n8k16(frag_a, frag_b0, frag_c[mb][0]);
            ptx_mma_m16n8k16(frag_a, frag_b1, frag_c[mb][1]);
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    #pragma unroll
    for (int mb = 0; mb < tile_m_blocks; ++mb)
    {
        int rows = min(16, size_m - (m_base + mb * 16));
        if (rows <= 0) break;
        int row0 = lane / 4;
        int row1 = row0 + 8;
        int col = (lane % 4) * 2;
        int n0 = warp * 16;
        if (row0 < rows)
        {
            float* dst0 = sh_c + row0 * FAT_TILE_N + n0 + col;
            dst0[0] = frag_c[mb][0][0];
            dst0[1] = frag_c[mb][0][1];
            dst0[8] = frag_c[mb][1][0];
            dst0[9] = frag_c[mb][1][1];
        }
        if (row1 < rows)
        {
            float* dst1 = sh_c + row1 * FAT_TILE_N + n0 + col;
            dst1[0] = frag_c[mb][0][2];
            dst1[1] = frag_c[mb][0][3];
            dst1[8] = frag_c[mb][1][2];
            dst1[9] = frag_c[mb][1][3];
        }
        __syncthreads();

        for (int row = warp; row < rows; row += 8)
        {
            fat_had_ff_128(
                sh_c + row * FAT_TILE_N,
                sh_c + row * FAT_TILE_N,
                svh + n_base);
        }
        __syncthreads();

        for (int i = t; i < rows * FAT_TILE_N; i += FAT_THREADS)
        {
            int row = i / FAT_TILE_N;
            int col_out = i % FAT_TILE_N;
            int source_row = m_base + mb * 16 + row;
            float value = sh_c[i];
            if constexpr (scatter)
            {
                int64_t destination = token_idx[source_row];
                value *= __half2float(route_weight[source_row]);
                float* destination_ptr =
                    out + destination * out_stride_n + out_n_base + col_out;
                if constexpr (atomic_scatter)
                    atomicAdd(destination_ptr, value);
                else
                    *destination_ptr += value;
            }
            else
            {
                out[source_row * out_stride_n + out_n_base + col_out] = value;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void grouped_swiglu_had_128(
    const float* __restrict__ gate_up,
    half* __restrict__ out,
    const half* __restrict__ down_suh,
    int row,
    int segment,
    float limit)
{
    int lane = threadIdx.x & 31;
    int col = segment * 128 + lane * 4;
    int input_row = row * 2 * 1024;
    float4 gate = reinterpret_cast<const float4*>(gate_up + input_row + col)[0];
    float4 up = reinterpret_cast<const float4*>(
        gate_up + input_row + 1024 + col)[0];

    gate.x = fminf(gate.x, limit);
    gate.y = fminf(gate.y, limit);
    gate.z = fminf(gate.z, limit);
    gate.w = fminf(gate.w, limit);
    up.x = fminf(fmaxf(up.x, -limit), limit);
    up.y = fminf(fmaxf(up.y, -limit), limit);
    up.z = fminf(fmaxf(up.z, -limit), limit);
    up.w = fminf(fmaxf(up.w, -limit), limit);

    gate.x = rounded_swiglu(gate.x, up.x);
    gate.y = rounded_swiglu(gate.y, up.y);
    gate.z = rounded_swiglu(gate.z, up.z);
    gate.w = rounded_swiglu(gate.w, up.w);

    half4 value;
    value.x = __floats2half2_rn(gate.x, gate.y);
    value.y = __floats2half2_rn(gate.z, gate.w);
    half4 scale = reinterpret_cast<const half4*>(down_suh + col)[0];
    value.x = __hmul2(value.x, scale.x);
    value.y = __hmul2(value.y, scale.y);

    float v0 = __half2float(__low2half(value.x));
    float v1 = __half2float(__high2half(value.x));
    float v2 = __half2float(__low2half(value.y));
    float v3 = __half2float(__high2half(value.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    value.x = __floats2half2_rn(h0 * HAD_SCALE, h1 * HAD_SCALE);
    value.y = __floats2half2_rn(h2 * HAD_SCALE, h3 * HAD_SCALE);
    reinterpret_cast<half4*>(out + row * 1024 + col)[0] = value;
}

// Build a compact M64 task list without exposing routing counts to the host.
// One thread is intentional: 288 scalar entries are cheaper than a scan setup,
// and this launch is asynchronous with respect to Python.
__global__ void exl3_fat_plan_kernel(
    const int64_t* __restrict__ offsets,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    int experts,
    int cap,
    int task_capacity)
{
    if (blockIdx.x || threadIdx.x) return;
    int next = 0;
    int scratch_base = 0;
    for (int expert = 0; expert < experts; ++expert)
    {
        int start = static_cast<int>(offsets[expert]);
        int count = static_cast<int>(offsets[expert + 1] - offsets[expert]);
        if (count <= cap) continue;
        for (int row = 0; row < count; row += FAT_TILE_M_SMALL)
        {
            if (next >= task_capacity) return;
            int32_t* task = tasks + 4 * next++;
            task[0] = expert;
            task[1] = start + row;
            task[2] = min(FAT_TILE_M_SMALL, count - row);
            task[3] = scratch_base + row;
        }
        scratch_base += count;
    }
    *task_count = next;
}

__global__ __launch_bounds__(FAT_THREADS)
void exl3_fat_grouped_gather_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ token_sorted,
    half* __restrict__ h13,
    const int64_t* __restrict__ gate_suh,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    int task_capacity)
{
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warps_grid = (gridDim.x * blockDim.x) / 32;
    constexpr int segments = 4096 / 128;
    constexpr int items_per_task = FAT_TILE_M_SMALL * segments;
    for (int item = warp_global; item < task_capacity * items_per_task;
         item += warps_grid)
    {
        int task_id = item / items_per_task;
        if (task_id >= *task_count) continue;
        int rem = item % items_per_task;
        int row = rem / segments;
        int seg = rem % segments;
        const int32_t* task = tasks + 4 * task_id;
        if (row >= task[2]) continue;
        int expert = task[0];
        int route = task[1] + row;
        int scratch_row = task[3] + row;
        int token = static_cast<int>(token_sorted[route]);
        const half* suh = reinterpret_cast<const half*>(gate_suh[expert]);
        had_hf_r_128_inner<true, false>(
            x + token * 4096 + seg * 128,
            h13 + scratch_row * 4096 + seg * 128,
            suh + seg * 128,
            HAD_SCALE);
    }
}

__global__ __launch_bounds__(FAT_THREADS)
void exl3_fat_grouped_gate_kernel(
    const half* __restrict__ h13,
    float* __restrict__ gate_up,
    const int64_t* __restrict__ gate_trellis,
    const int64_t* __restrict__ gate_svh,
    const int64_t* __restrict__ up_trellis,
    const int64_t* __restrict__ up_svh,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    int task_capacity)
{
    extern __shared__ unsigned char shared_raw[];
    constexpr int FAT_STAGES = 3;
    half* sh_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(
        sh_a + FAT_STAGES * FAT_TILE_M_SMALL * FAT_TILE_K);
    float* sh_c = reinterpret_cast<float*>(
        sh_b + FAT_STAGES * FAT_N_BLOCKS * FAT_PACKED_WORDS);
    constexpr int groups = 2 * 1024 / FAT_TILE_N;
    for (int item = blockIdx.x; item < task_capacity * groups; item += gridDim.x)
    {
        int task_id = item / groups;
        if (task_id >= *task_count) continue;
        int group_pair = item % groups;
        int which = group_pair / (1024 / FAT_TILE_N);
        int group = group_pair % (1024 / FAT_TILE_N);
        const int32_t* task = tasks + 4 * task_id;
        int expert = task[0];
        int rows = task[2];
        int scratch_row = task[3];
        const int64_t* tp = which ? up_trellis : gate_trellis;
        const int64_t* vp = which ? up_svh : gate_svh;
        grouped_fat_m64_tile<false>(
            h13 + static_cast<size_t>(scratch_row) * 4096,
            reinterpret_cast<const uint16_t*>(tp[expert]),
            gate_up + static_cast<size_t>(scratch_row) * 2048,
            reinterpret_cast<const half*>(vp[expert]),
            nullptr,
            nullptr,
            rows,
            4096,
            1024,
            2048,
            0,
            group * FAT_TILE_N,
            which * 1024 + group * FAT_TILE_N,
            sh_a,
            sh_b,
            sh_c);
    }
}

__global__ __launch_bounds__(FAT_THREADS)
void exl3_fat_grouped_activation_kernel(
    const float* __restrict__ gate_up,
    half* __restrict__ h2,
    const int64_t* __restrict__ down_suh,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    int task_capacity,
    float limit)
{
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warps_grid = (gridDim.x * blockDim.x) / 32;
    constexpr int segments = 1024 / 128;
    constexpr int items_per_task = FAT_TILE_M_SMALL * segments;
    for (int item = warp_global; item < task_capacity * items_per_task;
         item += warps_grid)
    {
        int task_id = item / items_per_task;
        if (task_id >= *task_count) continue;
        int rem = item % items_per_task;
        int row = rem / segments;
        int seg = rem % segments;
        const int32_t* task = tasks + 4 * task_id;
        if (row >= task[2]) continue;
        int expert = task[0];
        int scratch_row = task[3] + row;
        grouped_swiglu_had_128(
            gate_up + static_cast<size_t>(scratch_row) * 2048,
            h2 + static_cast<size_t>(scratch_row) * 1024,
            reinterpret_cast<const half*>(down_suh[expert]),
            0,
            seg,
            limit);
    }
}

__global__ __launch_bounds__(FAT_THREADS)
void exl3_fat_grouped_down_kernel(
    const half* __restrict__ h2,
    float* __restrict__ out,
    const int64_t* __restrict__ token_sorted,
    const half* __restrict__ weight_sorted,
    const int64_t* __restrict__ down_trellis,
    const int64_t* __restrict__ down_svh,
    const int32_t* __restrict__ tasks,
    const int32_t* __restrict__ task_count,
    int task_capacity)
{
    extern __shared__ unsigned char shared_raw[];
    constexpr int FAT_STAGES = 3;
    half* sh_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(
        sh_a + FAT_STAGES * FAT_TILE_M_SMALL * FAT_TILE_K);
    float* sh_c = reinterpret_cast<float*>(
        sh_b + FAT_STAGES * FAT_N_BLOCKS * FAT_PACKED_WORDS);
    constexpr int groups = 4096 / FAT_TILE_N;
    for (int item = blockIdx.x; item < task_capacity * groups; item += gridDim.x)
    {
        int task_id = item / groups;
        if (task_id >= *task_count) continue;
        int group = item % groups;
        const int32_t* task = tasks + 4 * task_id;
        int expert = task[0];
        int route = task[1];
        int rows = task[2];
        int scratch_row = task[3];
        grouped_fat_m64_tile<true, true>(
            h2 + static_cast<size_t>(scratch_row) * 1024,
            reinterpret_cast<const uint16_t*>(down_trellis[expert]),
            out,
            reinterpret_cast<const half*>(down_svh[expert]),
            token_sorted + route,
            weight_sorted + route,
            rows,
            1024,
            4096,
            4096,
            0,
            group * FAT_TILE_N,
            group * FAT_TILE_N,
            sh_a,
            sh_b,
            sh_c);
    }
}

void check_common(
    const at::Tensor& a,
    const at::Tensor& packed,
    const at::Tensor& out,
    const at::Tensor& svh,
    int64_t K,
    bool mcg,
    bool mul1)
{
    TORCH_CHECK(a.is_cuda() && packed.is_cuda() && out.is_cuda() && svh.is_cuda(),
                "exl3_fat_gemm tensors must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && packed.is_contiguous() && out.is_contiguous() && svh.is_contiguous(),
                "exl3_fat_gemm tensors must be contiguous");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(packed.scalar_type() == at::kShort, "packed must be int16");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(svh.scalar_type() == at::kHalf, "svh must be float16");
    TORCH_CHECK(a.dim() == 2 && packed.dim() == 3 && out.dim() == 2 && svh.dim() == 1,
                "exl3_fat_gemm expects rank-2/rank-3 tensors");
    TORCH_CHECK(K == 4 && mcg && !mul1,
                "exl3_fat_gemm currently supports only K4 MCG tensors");
    TORCH_CHECK(a.size(1) == packed.size(0) * 16,
                "a K dimension does not match packed tensor");
    TORCH_CHECK(svh.numel() == packed.size(1) * 16,
                "svh N dimension does not match packed tensor");
    TORCH_CHECK(svh.numel() % FAT_TILE_N == 0,
                "output dimension must be divisible by 128");
    TORCH_CHECK(packed.size(2) == FAT_PACKED_WORDS,
                "packed K4 block width must be 64 int16 words");
    TORCH_CHECK(a.device() == packed.device() && a.device() == out.device() && a.device() == svh.device(),
                "exl3_fat_gemm tensors must share a device");
}

template <int tile_m, bool scatter, bool paired = false>
void launch(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    at::Tensor packed_pair = at::Tensor(),
    at::Tensor svh_pair = at::Tensor())
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int size_m = static_cast<int>(a.size(0));
    int size_k = static_cast<int>(a.size(1));
    int size_n = static_cast<int>(svh.numel());
    dim3 block(FAT_THREADS);
    int out_stride_n = paired ? 2 * size_n : size_n;
    dim3 grid(out_stride_n / FAT_TILE_N, (size_m + tile_m - 1) / tile_m);
    // S2b: 3 pipeline stages of (A tile + B tile), plus the 16-row sh_c scratch.
    constexpr int FAT_STAGES = 3;
    size_t shared = FAT_STAGES * tile_m * FAT_TILE_K * sizeof(half)
                  + FAT_STAGES * FAT_N_BLOCKS * FAT_PACKED_WORDS * sizeof(uint16_t)
                  + 16 * FAT_TILE_N * sizeof(float);
    exl3_fat_gemm_kernel<tile_m, scatter, paired><<<grid, block, shared, stream>>>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const uint16_t*>(packed.data_ptr()),
        paired ? reinterpret_cast<const uint16_t*>(packed_pair.data_ptr()) : nullptr,
        reinterpret_cast<float*>(out.data_ptr()),
        reinterpret_cast<const half*>(svh.data_ptr()),
        paired ? reinterpret_cast<const half*>(svh_pair.data_ptr()) : nullptr,
        scatter ? reinterpret_cast<const int64_t*>(token_idx.data_ptr()) : nullptr,
        scatter ? reinterpret_cast<const half*>(route_weight.data_ptr()) : nullptr,
        size_m,
        size_k,
        size_n,
        out_stride_n);
    cuda_check(cudaPeekAtLastError());
}

}  // namespace

void exl3_fat_gemm(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == svh.numel(),
                "out shape must be [M, N]");
    launch<FAT_TILE_M, false>(a, packed, out, svh, at::Tensor(), at::Tensor());
}

void exl3_fat_gemm_m64(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == svh.numel(),
                "out shape must be [M, N]");
    launch<FAT_TILE_M_SMALL, false>(
        a, packed, out, svh, at::Tensor(), at::Tensor());
}

void exl3_fat_gemm_pair(
    at::Tensor a,
    at::Tensor packed_gate,
    at::Tensor packed_up,
    at::Tensor out,
    at::Tensor svh_gate,
    at::Tensor svh_up,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed_gate, out, svh_gate, K, mcg, mul1);
    check_common(a, packed_up, out, svh_up, K, mcg, mul1);
    TORCH_CHECK(packed_gate.sizes() == packed_up.sizes(),
                "paired packed tensors must have identical shapes");
    TORCH_CHECK(svh_gate.sizes() == svh_up.sizes(),
                "paired output scales must have identical shapes");
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == 2 * svh_gate.numel(),
                "paired out shape must be [M, 2N]");
    launch<FAT_TILE_M, false, true>(
        a,
        packed_gate,
        out,
        svh_gate,
        at::Tensor(),
        at::Tensor(),
        packed_up,
        svh_up);
}

void exl3_fat_gemm_pair_m64(
    at::Tensor a,
    at::Tensor packed_gate,
    at::Tensor packed_up,
    at::Tensor out,
    at::Tensor svh_gate,
    at::Tensor svh_up,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed_gate, out, svh_gate, K, mcg, mul1);
    check_common(a, packed_up, out, svh_up, K, mcg, mul1);
    TORCH_CHECK(packed_gate.sizes() == packed_up.sizes(),
                "paired packed tensors must have identical shapes");
    TORCH_CHECK(svh_gate.sizes() == svh_up.sizes(),
                "paired output scales must have identical shapes");
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == 2 * svh_gate.numel(),
                "paired out shape must be [M, 2N]");
    launch<FAT_TILE_M_SMALL, false, true>(
        a,
        packed_gate,
        out,
        svh_gate,
        at::Tensor(),
        at::Tensor(),
        packed_up,
        svh_up);
}

void exl3_fat_swiglu_had(
    at::Tensor gate_up,
    at::Tensor out,
    at::Tensor down_suh,
    double limit)
{
    TORCH_CHECK(gate_up.is_cuda() && out.is_cuda() && down_suh.is_cuda(),
                "exl3_fat_swiglu_had tensors must be CUDA tensors");
    TORCH_CHECK(gate_up.is_contiguous() && out.is_contiguous() && down_suh.is_contiguous(),
                "exl3_fat_swiglu_had tensors must be contiguous");
    TORCH_CHECK(gate_up.scalar_type() == at::kFloat, "gate_up must be float32");
    TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be float16");
    TORCH_CHECK(down_suh.scalar_type() == at::kHalf, "down_suh must be float16");
    TORCH_CHECK(gate_up.dim() == 2 && out.dim() == 2 && down_suh.dim() == 1,
                "exl3_fat_swiglu_had expects rank-2/rank-1 tensors");
    TORCH_CHECK(gate_up.size(0) == out.size(0), "row count mismatch");
    TORCH_CHECK(gate_up.size(1) == 2 * out.size(1), "gate_up must be [M, 2N]");
    TORCH_CHECK(down_suh.numel() == out.size(1), "down_suh N mismatch");
    TORCH_CHECK(out.size(1) % 128 == 0, "N must be divisible by 128");
    TORCH_CHECK(
        gate_up.device() == out.device() && gate_up.device() == down_suh.device(),
        "exl3_fat_swiglu_had tensors must share a device");

    const at::cuda::OptionalCUDAGuard device_guard(gate_up.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(gate_up.size(0), out.size(1) / 128);
    exl3_fat_swiglu_had_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const float*>(gate_up.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<const half*>(down_suh.data_ptr()),
        static_cast<int>(out.size(1)),
        static_cast<float>(limit));
    cuda_check(cudaPeekAtLastError());
}

void exl3_fat_gemm_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(1) == svh.numel(), "out N dimension mismatch");
    TORCH_CHECK(token_idx.is_cuda() && route_weight.is_cuda(),
                "routing tensors must be CUDA tensors");
    TORCH_CHECK(token_idx.is_contiguous() && route_weight.is_contiguous(),
                "routing tensors must be contiguous");
    TORCH_CHECK(token_idx.scalar_type() == at::kLong, "token_idx must be int64");
    TORCH_CHECK(route_weight.scalar_type() == at::kHalf, "route_weight must be float16");
    TORCH_CHECK(token_idx.numel() == a.size(0) && route_weight.numel() == a.size(0),
                "routing tensors must have M elements");
    TORCH_CHECK(token_idx.device() == a.device() && route_weight.device() == a.device(),
                "routing tensors must share a device with a");
    launch<FAT_TILE_M, true>(
        a, packed, out, svh, token_idx, route_weight);
}

void exl3_fat_gemm_scatter_m64(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(1) == svh.numel(), "out N dimension mismatch");
    TORCH_CHECK(token_idx.is_cuda() && route_weight.is_cuda(),
                "routing tensors must be CUDA tensors");
    TORCH_CHECK(token_idx.is_contiguous() && route_weight.is_contiguous(),
                "routing tensors must be contiguous");
    TORCH_CHECK(token_idx.scalar_type() == at::kLong, "token_idx must be int64");
    TORCH_CHECK(route_weight.scalar_type() == at::kHalf, "route_weight must be float16");
    TORCH_CHECK(token_idx.numel() == a.size(0) && route_weight.numel() == a.size(0),
                "routing tensors must have M elements");
    TORCH_CHECK(token_idx.device() == a.device() && route_weight.device() == a.device(),
                "routing tensors must share a device with a");
    launch<FAT_TILE_M_SMALL, true>(
        a, packed, out, svh, token_idx, route_weight);
}

void exl3_grouped_prefill_k4(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_offsets,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& had_input,
    const at::Tensor& gate_up,
    const at::Tensor& had_down,
    const at::Tensor& tasks,
    const at::Tensor& task_count,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    int64_t cap,
    double act_limit)
{
    const at::cuda::OptionalCUDAGuard guard(hidden_state.device());
    auto check = [&](const at::Tensor& t, at::ScalarType dtype, const char* name)
    {
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(t.device() == hidden_state.device(), name, " is on the wrong device");
    };
    check(hidden_state, at::kHalf, "hidden_state");
    check(output_state, at::kFloat, "output_state");
    check(expert_offsets, at::kLong, "expert_offsets");
    check(token_sorted, at::kLong, "token_sorted");
    check(weight_sorted, at::kHalf, "weight_sorted");
    check(had_input, at::kHalf, "had_input");
    check(gate_up, at::kFloat, "gate_up");
    check(had_down, at::kHalf, "had_down");
    check(tasks, at::kInt, "tasks");
    check(task_count, at::kInt, "task_count");
    check(gate_trellis, at::kLong, "gate_trellis");
    check(gate_suh, at::kLong, "gate_suh");
    check(gate_svh, at::kLong, "gate_svh");
    check(up_trellis, at::kLong, "up_trellis");
    check(up_svh, at::kLong, "up_svh");
    check(down_trellis, at::kLong, "down_trellis");
    check(down_suh, at::kLong, "down_suh");
    check(down_svh, at::kLong, "down_svh");

    TORCH_CHECK(hidden_state.dim() == 2 && hidden_state.size(1) == 4096,
                "grouped prefill requires hidden_state [tokens,4096]");
    TORCH_CHECK(output_state.sizes() == hidden_state.sizes(),
                "output_state shape mismatch");
    TORCH_CHECK(expert_offsets.dim() == 1 && expert_offsets.numel() >= 2,
                "expert_offsets must be [experts+1]");
    int experts = static_cast<int>(expert_offsets.numel() - 1);
    int tokens = static_cast<int>(hidden_state.size(0));
    TORCH_CHECK(token_sorted.numel() == weight_sorted.numel(),
                "routing tensors disagree");
    int routes = static_cast<int>(token_sorted.numel());
    TORCH_CHECK(had_input.dim() == 2 && had_input.size(0) >= routes
                && had_input.size(1) == 4096,
                "had_input scratch must be at least [routes,4096]");
    TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(0) >= routes
                && gate_up.size(1) == 2048,
                "gate_up scratch must be at least [routes,2048]");
    TORCH_CHECK(had_down.dim() == 2 && had_down.size(0) >= routes
                && had_down.size(1) == 1024,
                "had_down scratch must be at least [routes,1024]");
    TORCH_CHECK(tasks.dim() == 2 && tasks.size(1) == 4 && tasks.size(0) >= experts,
                "tasks must be [capacity,4]");
    TORCH_CHECK(task_count.numel() == 1, "task_count must contain one int32");
    for (const at::Tensor* table : {
             &gate_trellis, &gate_suh, &gate_svh, &up_trellis,
             &up_svh, &down_trellis, &down_suh, &down_svh})
    {
        TORCH_CHECK(table->numel() == experts,
                    "expert pointer table length mismatch");
    }
    TORCH_CHECK(cap >= 1 && cap < tokens, "cap must be in [1,tokens)");

    int device = 0;
    int sms = 0;
    int resident = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    constexpr int FAT_STAGES = 3;
    constexpr size_t shared =
        FAT_STAGES * FAT_TILE_M_SMALL * FAT_TILE_K * sizeof(half)
        + FAT_STAGES * FAT_N_BLOCKS * FAT_PACKED_WORDS * sizeof(uint16_t)
        + 16 * FAT_TILE_N * sizeof(float);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, exl3_fat_grouped_gate_kernel, FAT_THREADS, shared);
    TORCH_CHECK(resident > 0 && sms > 0,
                "grouped prefill kernel has zero occupancy");
    int grid_size = resident * sms;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const half* x = reinterpret_cast<const half*>(hidden_state.data_ptr<c10::Half>());
    float* out = output_state.data_ptr<float>();
    const int64_t* offsets = expert_offsets.data_ptr<int64_t>();
    const int64_t* sorted_tokens = token_sorted.data_ptr<int64_t>();
    const half* sorted_weights = reinterpret_cast<const half*>(
        weight_sorted.data_ptr<c10::Half>());
    half* hi = reinterpret_cast<half*>(had_input.data_ptr<c10::Half>());
    float* go = gate_up.data_ptr<float>();
    half* hd = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    int32_t* task_data = tasks.data_ptr<int32_t>();
    int32_t* task_count_data = task_count.data_ptr<int32_t>();
    const int64_t* gt = gate_trellis.data_ptr<int64_t>();
    const int64_t* gu = gate_suh.data_ptr<int64_t>();
    const int64_t* gv = gate_svh.data_ptr<int64_t>();
    const int64_t* ut = up_trellis.data_ptr<int64_t>();
    const int64_t* uv = up_svh.data_ptr<int64_t>();
    const int64_t* dt = down_trellis.data_ptr<int64_t>();
    const int64_t* du = down_suh.data_ptr<int64_t>();
    const int64_t* dv = down_svh.data_ptr<int64_t>();
    int launch_experts = experts;
    int launch_cap = static_cast<int>(cap);
    int task_capacity = static_cast<int>(tasks.size(0));
    float limit = static_cast<float>(act_limit);

    exl3_fat_plan_kernel<<<1, 1, 0, stream>>>(
        offsets, task_data, task_count_data, launch_experts, launch_cap,
        task_capacity);
    exl3_fat_grouped_gather_kernel<<<grid_size, FAT_THREADS, 0, stream>>>(
        x, sorted_tokens, hi, gu, task_data, task_count_data, task_capacity);
    exl3_fat_grouped_gate_kernel<<<grid_size, FAT_THREADS, shared, stream>>>(
        hi, go, gt, gv, ut, uv, task_data, task_count_data, task_capacity);
    exl3_fat_grouped_activation_kernel<<<grid_size, FAT_THREADS, 0, stream>>>(
        go, hd, du, task_data, task_count_data, task_capacity, limit);
    exl3_fat_grouped_down_kernel<<<grid_size, FAT_THREADS, shared, stream>>>(
        hd, out, sorted_tokens, sorted_weights, dt, dv,
        task_data, task_count_data, task_capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
