// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Cooperative GLM-5.3 EXL3 decode experiment for the exact two-Spark shape:
// hidden=4096, rank-local expert intermediate=1024, K4 MCG, top-8.
//
// The register-streaming trellis decode and MMA arrangement are derived from
// ExLlamaV3 and the cooperative p2b prototype in vcruz305/vllm-exl3. This
// implementation changes the execution contract substantially: expert-sorted
// multi-row tiles, persistent caller-owned scratch, FP32 routed accumulation,
// GLM clamp semantics, and one CUDA-graph-safe launch for the whole layer.

#include <algorithm>
#include <cstdint>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "exl3_decode_moe.cuh"
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

namespace cg = cooperative_groups;

namespace {

constexpr int kHidden = 4096;
constexpr int kIntermediate = 1024;
constexpr int kRowsPerTile = 8;
constexpr int kThreads = 512;
constexpr int kWarps = 16;
constexpr int kCols = 32;
constexpr int kBits = 4;
constexpr float kHadScale = 0.088388347648f;

// The following three helpers retain ExLlamaV3's half-accumulating K4 decode
// math. Keeping the bit extraction and MMA arrangement unchanged makes the
// experiment about scheduling rather than a new quantization implementation.
__device__ __forceinline__ void mma_ab_h(
    const FragB& a01, const FragB& a23, const FragB& b, FragC_h& c) {
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
          "r"(bb[0]), "r"(bb[1]));
}

__device__ __forceinline__ void decode8_mcg(
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7,
    FragB& f0, FragB& f1) {
    f0[0] = decode_3inst_2<1>(w0, w1);
    f0[1] = decode_3inst_2<1>(w2, w3);
    f1[0] = decode_3inst_2<1>(w4, w5);
    f1[1] = decode_3inst_2<1>(w6, w7);
}

__device__ __forceinline__ void dq8_k4_mcg(
    uint32_t a, uint32_t b, FragB& f0, FragB& f1) {
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    decode8_mcg(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// One block computes 32 output columns for as many as eight rows of one
// expert. Sixteen warps split K; weights are fetched once per warp and reused
// across all live rows in the tile.
__device__ __forceinline__ void run_k4_tile(
    const uint16_t* __restrict__ trellis,
    const half* __restrict__ input,
    half* __restrict__ output,
    int rows,
    int size_k,
    int size_n,
    int group,
    float (*sh_red)[kRowsPerTile][kCols]) {
    constexpr int kNtilesPerWarp = 2;
    constexpr int kPrefetch = 4;
    constexpr int kFold = 4;
    constexpr int kWords = 8 * kBits;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int chunk = (kslices + kWarps - 1) / kWarps;
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const size_t slice_stride = static_cast<size_t>(ntiles) * kWords;
    const uint32_t* b32 = reinterpret_cast<const uint32_t*>(trellis);
    const uint32_t* bp = b32 + static_cast<size_t>(ks0) * slice_stride
        + group * kNtilesPerWarp * kWords + lane;
    const half2* a2 = reinterpret_cast<const half2*>(input);
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int row = lane >> 2;
    const bool row_ok = row < rows;
    const size_t a_row = static_cast<size_t>(row) * (size_k / 2);

    auto load_b = [&](int i, int n) -> uint32_t {
        return __ldcs(bp + static_cast<size_t>(i) * slice_stride + n * 32);
    };

    uint32_t pref[kPrefetch][kNtilesPerWarp] = {};
#pragma unroll
    for (int d = 0; d < kPrefetch; ++d) {
        if (d < myn) {
#pragma unroll
            for (int n = 0; n < kNtilesPerWarp; ++n) pref[d][n] = load_b(d, n);
        }
    }

    FragC_h frag[kNtilesPerWarp][2] = {};
    float2 accum[kNtilesPerWarp][2] = {};
    for (int ib = 0; ib < myn; ib += kPrefetch) {
#pragma unroll
        for (int d = 0; d < kPrefetch; ++d) {
            const int i = ib + d;
            if (i >= myn) break;
            uint32_t words[kNtilesPerWarp];
#pragma unroll
            for (int n = 0; n < kNtilesPerWarp; ++n) words[n] = pref[d][n];
            if (i + kPrefetch < myn) {
#pragma unroll
                for (int n = 0; n < kNtilesPerWarp; ++n)
                    pref[d][n] = load_b(i + kPrefetch, n);
            }

            const size_t a_col = static_cast<size_t>(ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = row_ok ? a2[a_row + a_col] : hzero;
            a23[0] = row_ok ? a2[a_row + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;
#pragma unroll
            for (int n = 0; n < kNtilesPerWarp; ++n) {
                FragB f0, f1;
                const uint32_t aw = __shfl_sync(
                    0xffffffffu, words[n], (lane + 31) & 31);
                dq8_k4_mcg(aw, words[n], f0, f1);
                mma_ab_h(a01, a23, f0, frag[n][0]);
                mma_ab_h(a01, a23, f1, frag[n][1]);
            }
            if ((d + 1) % kFold == 0 || i + 1 == myn) {
#pragma unroll
                for (int n = 0; n < kNtilesPerWarp; ++n) {
#pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        accum[n][f].x += __low2float(frag[n][f][0]);
                        accum[n][f].y += __high2float(frag[n][f][0]);
                        frag[n][f][0] = hzero;
                    }
                }
            }
        }
    }

    if (row_ok) {
        const int c0 = 2 * (lane & 3);
#pragma unroll
        for (int n = 0; n < kNtilesPerWarp; ++n) {
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                const int col = n * 16 + f * 8 + c0;
                sh_red[warp][row][col] = accum[n][f].x;
                sh_red[warp][row][col + 1] = accum[n][f].y;
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows * kCols; idx += kThreads) {
        const int r = idx / kCols;
        const int col_local = idx % kCols;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) sum += sh_red[w][r][col_local];
        output[static_cast<size_t>(r) * size_n + group * kCols + col_local]
            = __float2half_rn(sum);
    }
    __syncthreads();
}

__global__ __launch_bounds__(kThreads)
void decode_moe_k4_kernel(
    const half* __restrict__ x,
    float* __restrict__ output,
    const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ expert_sorted,
    const int64_t* __restrict__ token_sorted,
    const half* __restrict__ weight_sorted,
    half* __restrict__ had_gate,
    half* __restrict__ had_up,
    half* __restrict__ gate,
    half* __restrict__ up,
    half* __restrict__ had_down,
    half* __restrict__ down,
    const int64_t* __restrict__ gt,
    const int64_t* __restrict__ gu,
    const int64_t* __restrict__ gv,
    const int64_t* __restrict__ ut,
    const int64_t* __restrict__ uu,
    const int64_t* __restrict__ uv,
    const int64_t* __restrict__ dt,
    const int64_t* __restrict__ du,
    const int64_t* __restrict__ dv,
    int tokens,
    int routes,
    int experts,
    float act_limit) {
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threads = gridDim.x * blockDim.x;
    const int warp_global = tid / 32;
    const int warps_grid = threads / 32;
    const int lane = threadIdx.x & 31;
    __shared__ float sh_red[kWarps][kRowsPerTile][kCols];

    for (int i = tid; i < tokens * kHidden; i += threads) output[i] = 0.0f;
    grid.sync();

    // Input Hadamard and per-expert input scales for every sorted route.
    for (int item = warp_global; item < routes * (kHidden / 128);
         item += warps_grid) {
        const int route = item / (kHidden / 128);
        const int seg = item % (kHidden / 128);
        const int expert = expert_sorted[route];
        const int token = static_cast<int>(token_sorted[route]);
        const half* gsu = reinterpret_cast<const half*>(gu[expert]);
        const half* usu = reinterpret_cast<const half*>(uu[expert]);
        had_hf_r_128_inner<true, false>(
            x + token * kHidden + seg * 128,
            had_gate + route * kHidden + seg * 128,
            gsu + seg * 128, kHadScale);
        had_hf_r_128_inner<true, false>(
            x + token * kHidden + seg * 128,
            had_up + route * kHidden + seg * 128,
            usu + seg * 128, kHadScale);
    }
    grid.sync();

    const int row_tiles = (tokens + kRowsPerTile - 1) / kRowsPerTile;
    const int gu_groups = kIntermediate / kCols;
    const int gu_items = experts * row_tiles * 2 * gu_groups;
    for (int item = blockIdx.x; item < gu_items; item += gridDim.x) {
        int q = item;
        const int group = q % gu_groups; q /= gu_groups;
        const int which = q & 1; q >>= 1;
        const int tile = q % row_tiles;
        const int expert = q / row_tiles;
        const int start = static_cast<int>(offsets[expert]);
        const int count = static_cast<int>(offsets[expert + 1] - offsets[expert]);
        const int tile_start = tile * kRowsPerTile;
        const int rows = min(kRowsPerTile, count - tile_start);
        if (rows > 0) {
            const int route = start + tile_start;
            const int64_t* tp = which ? ut : gt;
            run_k4_tile(
                reinterpret_cast<const uint16_t*>(tp[expert]),
                (which ? had_up : had_gate) + route * kHidden,
                (which ? up : gate) + route * kIntermediate,
                rows, kHidden, kIntermediate, group, sh_red);
        }
    }
    grid.sync();

    // Fused output Hadamards, exact GLM clamp/SwiGLU, and down-input Hadamard.
    for (int item = warp_global; item < routes * (kIntermediate / 128);
         item += warps_grid) {
        const int route = item / (kIntermediate / 128);
        const int seg = item % (kIntermediate / 128);
        const int expert = expert_sorted[route];
        const half* gsv = reinterpret_cast<const half*>(gv[expert]);
        const half* usv = reinterpret_cast<const half*>(uv[expert]);
        const half* dsu = reinterpret_cast<const half*>(du[expert]);
        had_hf_r_128_guad_inner(
            gate + route * kIntermediate + seg * 128,
            up + route * kIntermediate + seg * 128,
            had_down + route * kIntermediate + seg * 128,
            gsv + seg * 128, usv + seg * 128, dsu + seg * 128,
            kHadScale, act_limit, 0);
    }
    grid.sync();

    const int down_groups = kHidden / kCols;
    const int down_items = experts * row_tiles * down_groups;
    for (int item = blockIdx.x; item < down_items; item += gridDim.x) {
        int q = item;
        const int group = q % down_groups; q /= down_groups;
        const int tile = q % row_tiles;
        const int expert = q / row_tiles;
        const int start = static_cast<int>(offsets[expert]);
        const int count = static_cast<int>(offsets[expert + 1] - offsets[expert]);
        const int tile_start = tile * kRowsPerTile;
        const int rows = min(kRowsPerTile, count - tile_start);
        if (rows > 0) {
            const int route = start + tile_start;
            run_k4_tile(
                reinterpret_cast<const uint16_t*>(dt[expert]),
                had_down + route * kIntermediate,
                down + route * kHidden,
                rows, kIntermediate, kHidden, group, sh_red);
        }
    }
    grid.sync();

    // Output Hadamard, routing weight and FP32 atomic combination. This is the
    // same accumulation precision as the production ExLlamaV3 path.
    for (int item = warp_global; item < routes * (kHidden / 128);
         item += warps_grid) {
        const int route = item / (kHidden / 128);
        const int seg = item % (kHidden / 128);
        const int expert = expert_sorted[route];
        const int token = static_cast<int>(token_sorted[route]);
        const half* dsv = reinterpret_cast<const half*>(dv[expert]);
        const float route_scale = kHadScale * __half2float(weight_sorted[route]);
        had_hf_r_128_d_inner(
            down + route * kHidden + seg * 128,
            output + token * kHidden + seg * 128,
            dsv + seg * 128, route_scale);
    }
}

void check_cuda(const at::Tensor& t, at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void exl3_decode_moe_k4(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_offsets,
    const at::Tensor& expert_sorted,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& had_gate,
    const at::Tensor& had_up,
    const at::Tensor& gate,
    const at::Tensor& up,
    const at::Tensor& had_down,
    const at::Tensor& down,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    double act_limit) {
    const at::cuda::OptionalCUDAGuard guard(hidden_state.device());
    check_cuda(hidden_state, at::kHalf, "hidden_state");
    check_cuda(output_state, at::kFloat, "output_state");
    check_cuda(expert_offsets, at::kLong, "expert_offsets");
    check_cuda(expert_sorted, at::kInt, "expert_sorted");
    check_cuda(token_sorted, at::kLong, "token_sorted");
    check_cuda(weight_sorted, at::kHalf, "weight_sorted");
    check_cuda(had_gate, at::kHalf, "had_gate");
    check_cuda(had_up, at::kHalf, "had_up");
    check_cuda(gate, at::kHalf, "gate");
    check_cuda(up, at::kHalf, "up");
    check_cuda(had_down, at::kHalf, "had_down");
    check_cuda(down, at::kHalf, "down");
    TORCH_CHECK(hidden_state.dim() == 2 && hidden_state.size(1) == kHidden,
                "decode kernel requires [tokens,4096]");
    TORCH_CHECK(output_state.sizes() == hidden_state.sizes(),
                "output shape mismatch");
    const int tokens = static_cast<int>(hidden_state.size(0));
    const int routes = static_cast<int>(token_sorted.numel());
    const int experts = static_cast<int>(expert_offsets.numel() - 1);
    TORCH_CHECK(tokens >= 1 && tokens <= 32, "decode kernel supports 1..32 tokens");
    TORCH_CHECK(routes == expert_sorted.numel() && routes == weight_sorted.numel(),
                "sorted routing tensors disagree");
    TORCH_CHECK(routes <= had_gate.size(0) && routes <= had_up.size(0)
                && routes <= gate.size(0) && routes <= up.size(0)
                && routes <= had_down.size(0) && routes <= down.size(0),
                "persistent scratch capacity is too small");
    TORCH_CHECK(had_gate.size(1) == kHidden && had_up.size(1) == kHidden
                && down.size(1) == kHidden, "hidden scratch shape mismatch");
    TORCH_CHECK(gate.size(1) == kIntermediate && up.size(1) == kIntermediate
                && had_down.size(1) == kIntermediate,
                "intermediate scratch shape mismatch");

    int device = 0;
    int sms = 0;
    int resident = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    constexpr int dynamic_smem = kWarps * 128 * sizeof(float);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, decode_moe_k4_kernel, kThreads, dynamic_smem);
    TORCH_CHECK(resident > 0 && sms > 0, "cooperative decode kernel has zero occupancy");
    const int grid_size = resident * sms;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const half* x = reinterpret_cast<const half*>(hidden_state.data_ptr<c10::Half>());
    float* out = output_state.data_ptr<float>();
    const int64_t* offsets = expert_offsets.data_ptr<int64_t>();
    const int32_t* experts_sorted = expert_sorted.data_ptr<int32_t>();
    const int64_t* tokens_sorted = token_sorted.data_ptr<int64_t>();
    const half* weights = reinterpret_cast<const half*>(weight_sorted.data_ptr<c10::Half>());
    half* hg = reinterpret_cast<half*>(had_gate.data_ptr<c10::Half>());
    half* hu = reinterpret_cast<half*>(had_up.data_ptr<c10::Half>());
    half* g = reinterpret_cast<half*>(gate.data_ptr<c10::Half>());
    half* u = reinterpret_cast<half*>(up.data_ptr<c10::Half>());
    half* hd = reinterpret_cast<half*>(had_down.data_ptr<c10::Half>());
    half* d = reinterpret_cast<half*>(down.data_ptr<c10::Half>());
    const int64_t* gt = gate_trellis.data_ptr<int64_t>();
    const int64_t* gu = gate_suh.data_ptr<int64_t>();
    const int64_t* gv = gate_svh.data_ptr<int64_t>();
    const int64_t* ut = up_trellis.data_ptr<int64_t>();
    const int64_t* uu = up_suh.data_ptr<int64_t>();
    const int64_t* uv = up_svh.data_ptr<int64_t>();
    const int64_t* dt = down_trellis.data_ptr<int64_t>();
    const int64_t* du = down_suh.data_ptr<int64_t>();
    const int64_t* dv = down_svh.data_ptr<int64_t>();
    int launch_tokens = tokens;
    int launch_routes = routes;
    int launch_experts = experts;
    float limit = static_cast<float>(act_limit);

    void* args[] = {
        &x, &out, &offsets, &experts_sorted, &tokens_sorted, &weights,
        &hg, &hu, &g, &u, &hd, &d,
        &gt, &gu, &gv, &ut, &uu, &uv, &dt, &du, &dv,
        &launch_tokens, &launch_routes, &launch_experts, &limit,
    };
    cuda_check(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(decode_moe_k4_kernel), dim3(grid_size),
        dim3(kThreads), args, dynamic_smem, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
