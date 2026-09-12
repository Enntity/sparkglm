// SPDX-License-Identifier: AGPL-3.0-only

// GLM-only fused gate/up M16/N128/K64 tile; default-off host dispatch.
// Included by moe_w4a16_grouped_gemm.cu after its native-FP4 primitives.
// Four warps partition N into32 columns each; all128 threads still stage B.
// Arithmetic is shared with the standalone test-only down wrappers; the only
// production exports below accept gate/up N2048/K4096 and a real token gather.
#pragma once

template<bool PQ_VEC_SCALES>
__device__ __forceinline__ void glm_moe_m16n128_impl(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    unsigned int work_expert_id,
    unsigned int work_m_tile,
    unsigned int work_n_tile
) {
    // Standalone exact shapes only: host also validates row/gather capacities.
    // Every early return is CTA-uniform and precedes barriers.
    if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1
        || !((N == 4096 && K == 2048 && sorted_token_ids == nullptr)
             || (N == 2048 && K == 4096 && sorted_token_ids != nullptr))) return;
    const unsigned int expert_id = work_expert_id;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0 || M_expert > 5) return;

    const int cta_m_local = work_m_tile * 16;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = work_n_tile * N_TILE_LG;
    const unsigned char* B_expert =
        (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert =
        (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];
    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = 0;
    const unsigned int warp_n_offset = warp_id * 32;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // A is already compact FP4. Keep each row 16-byte aligned because the
    // loader issues 16-byte cp.async operations for both halves of the row.
    // (A 36-byte stride faults as CUDA_ERROR_MISALIGNED_ADDRESS on row 1.)
    __shared__ unsigned char m16_Ap[2][16][K_STEP_T64 / 2 + 16];
    __shared__ unsigned char m16_As[2][16][K_STEP_T64 / GROUP_SIZE];
    // B checkpoint layout is [K/2,N]; load it coalesced, then transpose the
    // 32-byte K run required by the native block-scaled MMA.
    __shared__ unsigned char m16_BpT[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char m16_Bp[N_TILE_LG][K_STEP_T64 / 2 + 16];
    __shared__ unsigned char m16_Bs[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG];
    __shared__ int m16_tok[16];

    if (threadIdx.x < 16) {
        const int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            m16_tok[local_row] = sorted_token_ids[cta_m + local_row];
        else
            m16_tok[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int num_groups = K / GROUP_SIZE;

    #define PQ4_ISSUE_LOADS(buf, kb) do { \
        if (threadIdx.x < 32) { \
            /* First32 loaders:16 rows x32 packed K bytes; all128 load B. */ \
            unsigned int row = threadIdx.x >> 1; \
            unsigned int col = (threadIdx.x & 1) << 4; \
            bool valid = (cta_m_local + row) < M_eff && ((kb) + col * 2 + 31 < K); \
            unsigned int a_row = (unsigned int)m16_tok[row]; \
            moe_cp_async_pred_16(&m16_Ap[(buf)][row][col], \
                &A_packed[(unsigned long long)a_row * (K / 2) + (kb) / 2 + col], \
                valid); \
        } \
        if constexpr (PQ_VEC_SCALES) { \
            /* One aligned 4-byte transaction loads all four A scales/row. */ \
            if (threadIdx.x < 16) { \
                unsigned int row = threadIdx.x; \
                bool valid = (cta_m_local + row) < M_eff; \
                unsigned int a_row = (unsigned int)m16_tok[row]; \
                moe_cp_async_pred_4(&m16_As[(buf)][row][0], \
                    &A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) \
                        + (kb) / GROUP_SIZE], valid); \
            } \
        } else { \
            /* Conservative scalar scale loader. */ \
            _Pragma("unroll") \
            for (int job = 0; job < 1; job++) { \
                unsigned int jid = threadIdx.x + job * 128; \
                if (jid >= 64) continue; \
                unsigned int row = jid >> 2; \
                unsigned int grp = jid & 3; \
                bool valid = (cta_m_local + row) < M_eff; \
                unsigned int a_row = (unsigned int)m16_tok[row]; \
                m16_As[(buf)][row][grp] = valid \
                    ? A_scale[(unsigned long long)a_row * (K / GROUP_SIZE) \
                        + (kb) / GROUP_SIZE + grp] \
                    : 0; \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                moe_cp_async_pred_16(&m16_BpT[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
            } \
        } \
        if constexpr (PQ_VEC_SCALES) { \
            /* 32 aligned 16-byte transactions replace 512 scalar copies. */ \
            if (threadIdx.x < 32) { \
                unsigned int g = threadIdx.x >> 3; \
                unsigned int ns = (threadIdx.x & 7) << 4; \
                unsigned int sg = (kb) / GROUP_SIZE + g; \
                unsigned int gns = cta_n + ns; \
                moe_cp_async_pred_16(&m16_Bs[(buf)][g][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N) && (sg < num_groups)); \
            } \
        } else { \
            unsigned int g = threadIdx.x >> 5; \
            unsigned int nn = threadIdx.x & 31; \
            unsigned int sg = (kb) / GROUP_SIZE + g; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int n_cur = rnd * 32 + nn; \
                unsigned int gns = cta_n + n_cur; \
                bool valid = (gns < N) && (sg < num_groups); \
                m16_Bs[(buf)][g][n_cur] = valid \
                    ? S_expert[(unsigned long long)sg * N + gns] : 0; \
            } \
        } \
    } while(0)

    #define PQ4_TRANSPOSE(buf) do { \
        unsigned int my_n = threadIdx.x; \
        _Pragma("unroll") \
        for (int q = 0; q < (K_STEP_T64 / 2) / 4; q++) { \
            unsigned int w = (unsigned int)m16_BpT[(buf)][q * 4 + 0][my_n] \
                | ((unsigned int)m16_BpT[(buf)][q * 4 + 1][my_n] << 8) \
                | ((unsigned int)m16_BpT[(buf)][q * 4 + 2][my_n] << 16) \
                | ((unsigned int)m16_BpT[(buf)][q * 4 + 3][my_n] << 24); \
            *(unsigned int*)&m16_Bp[my_n][q * 4] = w; \
        } \
    } while(0)

    #define PQ4_FRAG(P, ROW, KK) (*(const unsigned int*)&(P)[(ROW)][(KK) / 2])
    #define PQ4_COMPUTE_MMA(a_buf, b_buf) do { \
        unsigned int ra = warp_m_offset + group_id; \
        unsigned int a0 = PQ4_FRAG(m16_Ap[(a_buf)], ra,     tid * 8); \
        unsigned int a1 = PQ4_FRAG(m16_Ap[(a_buf)], ra + 8, tid * 8); \
        unsigned int a2 = PQ4_FRAG(m16_Ap[(a_buf)], ra,     32 + tid * 8); \
        unsigned int a3 = PQ4_FRAG(m16_Ap[(a_buf)], ra + 8, 32 + tid * 8); \
        unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2); \
        unsigned int sfa = (unsigned int)m16_As[(a_buf)][warp_m_offset + sfa_m][0] \
            | ((unsigned int)m16_As[(a_buf)][warp_m_offset + sfa_m][1] << 8) \
            | ((unsigned int)m16_As[(a_buf)][warp_m_offset + sfa_m][2] << 16) \
            | ((unsigned int)m16_As[(a_buf)][warp_m_offset + sfa_m][3] << 24); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 4; nt++) { \
            unsigned int nc = warp_n_offset + nt * 8 + group_id; \
            unsigned int b0 = PQ4_FRAG(m16_Bp, nc, tid * 8); \
            unsigned int b1 = PQ4_FRAG(m16_Bp, nc, 32 + tid * 8); \
            unsigned int sfn = warp_n_offset + nt * 8 + (lane_id >> 2); \
            unsigned int sfb = (unsigned int)m16_Bs[(b_buf)][0][sfn] \
                | ((unsigned int)m16_Bs[(b_buf)][1][sfn] << 8) \
                | ((unsigned int)m16_Bs[(b_buf)][2][sfn] << 16) \
                | ((unsigned int)m16_Bs[(b_buf)][3][sfn] << 24); \
            unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]), \
                 "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb),"h"(bidB),"h"(tidB_)); \
        } \
    } while(0)

    PQ4_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    PQ4_TRANSPOSE(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        PQ4_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        PQ4_COMPUTE_MMA(cur, cur);
        moe_cp_async_wait_all();
        __syncthreads();
        PQ4_TRANSPOSE(nxt);
        __syncthreads();
        cur = nxt;
    }
    PQ4_COMPUTE_MMA(cur, cur);

    #undef PQ4_ISSUE_LOADS
    #undef PQ4_TRANSPOSE
    #undef PQ4_FRAG
    #undef PQ4_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        unsigned int c0 = cta_n + warp_n_offset + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0] * scale2);
        if (r0v && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1] * scale2);
        if (r1v && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2] * scale2);
        if (r1v && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3] * scale2);
    }
}


// Projection-multiplexed gate/up: unchanged compact-map contract, now with
// actual token-major A gather. No extra builder, weight copy or down dispatch.
#define GLM_GU_M16_ARGS \
    const unsigned char* A_packed, const unsigned char* A_scale, \
    const unsigned long long* gate_packed_ptrs, const unsigned long long* gate_scale_ptrs, \
    const float* gate_scale2_vals, __nv_bfloat16* C_gate, \
    const unsigned long long* up_packed_ptrs, const unsigned long long* up_scale_ptrs, \
    const float* up_scale2_vals, __nv_bfloat16* C_up, \
    const int* expert_offsets, const int* sorted_token_ids, \
    unsigned int num_experts, unsigned int N, unsigned int K, \
    const unsigned int* worklist, const int* total_tiles, unsigned int max_tiles

template<bool VEC>
__device__ __forceinline__ void glm_moe_gate_up_m16_impl(GLM_GU_M16_ARGS) {
    if (blockIdx.y > 1 || N != 2048 || K != 4096 || !sorted_token_ids) return;
    const int raw_total = *total_tiles;
    const unsigned int count = raw_total > 0 ? min((unsigned int)raw_total, max_tiles) : 0u;
    const unsigned int wid = blockIdx.x;
    if (wid >= count) return;
    const unsigned int expert = worklist[wid * 2];
    const unsigned int packed = worklist[wid * 2 + 1];
    const unsigned int mt = packed >> 6, nt = packed & 0x3fu;
    // Host permits at most5 rows/expert: only the first M tile is legal.
    if (expert >= num_experts || mt != 0 || nt >= N / 128) return;
    const bool is_up = blockIdx.y != 0;
    glm_moe_m16n128_impl<VEC>(A_packed, A_scale,
        is_up ? up_packed_ptrs : gate_packed_ptrs,
        is_up ? up_scale_ptrs : gate_scale_ptrs,
        is_up ? up_scale2_vals : gate_scale2_vals,
        is_up ? C_up : C_gate, expert_offsets, sorted_token_ids, num_experts,
        N, K, expert, mt, nt);
}

#define GLM_GU_M16_CALL(VEC) \
    glm_moe_gate_up_m16_impl<VEC>(A_packed, A_scale, gate_packed_ptrs, gate_scale_ptrs, \
        gate_scale2_vals, C_gate, up_packed_ptrs, up_scale_ptrs, up_scale2_vals, C_up, \
        expert_offsets, sorted_token_ids, num_experts, N, K, worklist, total_tiles, max_tiles)

extern "C" __global__ void glm_moe_gate_up_m16n128(GLM_GU_M16_ARGS) {
    GLM_GU_M16_CALL(false);
}
extern "C" __global__ void glm_moe_gate_up_m16n128_vecscale(GLM_GU_M16_ARGS) {
    GLM_GU_M16_CALL(true);
}
#undef GLM_GU_M16_ARGS
#undef GLM_GU_M16_CALL
