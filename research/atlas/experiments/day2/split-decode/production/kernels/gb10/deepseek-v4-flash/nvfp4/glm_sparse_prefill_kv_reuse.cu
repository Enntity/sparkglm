// SPDX-License-Identifier: AGPL-3.0-only
// Opt-in bank-padded K=V specialization of the registered GLM tensor-core prefill kernel.
// Adapted from kernels/gb10/common/prefill_paged_compute_512.cuh (AGPL-3.0).
// Each CTA owns one query token and up to32 HEADS sharing its exact selected IDs.
// QK/PV use BF16 tensor cores with FP32 accumulation and BF16 probabilities.
// Requires identical K/V bytes. Grid=(ceil(heads/32),rows), block256, shared69376 bytes. Shared Q/K row stride is 520 BF16 values.
#include <cuda_bf16.h>
#include <math.h>

#undef LOAD_KV_TILE_512
#define LOAD_KV_TILE_512(cache, bt, smem_ptr, kv_s, kv_l, t, stride) \
    do { \
        for (unsigned int _i=(t); _i<TILE_CHUNKS_512; _i+=(stride)) { \
            const unsigned int _row=_i/64, _col=(_i%64)*8; \
            const unsigned int _selected=(kv_s)+_row; \
            const int _token=_selected<(kv_l) ? indices[_selected] : -1; \
            if (_token>=0) { \
                const unsigned int _physical=(bt)[(unsigned int)_token/cache_block_size]; \
                const unsigned long long _offset=((unsigned long long)_physical*cache_block_size+(unsigned int)_token%cache_block_size)*512+_col; \
                glm_kvp_cp16(&(smem_ptr)[_row*GLM_KVP_STRIDE+_col], (cache)+_offset); \
            } else { *((uint4*)&(smem_ptr)[_row*GLM_KVP_STRIDE+_col])=make_uint4(0,0,0,0); } \
        } \
    } while(0)

// Async global→shared 16-byte copy helpers (cp.async on NVIDIA + SCALE).
// The strix-hip copy of this header degrades these to synchronous uint4
// copies (AMD has no cp.async). Per-tree behavior comes purely from which
// header is compiled — no #if at the call sites.
__device__ __forceinline__ void glm_kvp_cp16(void* smem_dst, const void* gmem_src) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(_s), "l"(gmem_src));
}
__device__ __forceinline__ void glm_kvp_cp16_pred(void* smem_dst, const void* gmem_src, bool pred) {
    unsigned _s = __cvta_generic_to_shared(smem_dst);
    unsigned _b = pred ? 16u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(_s), "l"(gmem_src), "r"(_b));
}
__device__ __forceinline__ void glm_kvp_cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void glm_kvp_cp_wait()   { asm volatile("cp.async.wait_group 0;"); }

// Phase 2b precision fix (2026-05-24): the degree-3 Taylor polynomial
// here has up to 0.5% relative error at tf~1 (verified numerically vs
// torch.exp). For HDIM=512 (Gemma-4 long-attn), each softmax row spans
// hundreds of K-tile chunks, compounding the per-call error into
// measurable cosine drift. See sister fix in
// `prefill_paged_compute.cuh::sw_exp` for the Qwen3.6 HDIM=256 path.
// Default to accurate `__expf` (~2 ULP); polynomial available via
// `ATLAS_FAST_SOFTMAX_EXP` opt-in.
__device__ __forceinline__ float glm_kvp_exp(float x) {
#ifdef ATLAS_FAST_SOFTMAX_EXP
    float t = x * 1.4426950408889634f;
    float ti = floorf(t);
    float tf = t - ti;
    float p = 1.0f + tf * (0.6931471805599453f +
              tf * (0.2402265069591007f +
              tf * 0.05550410866482158f));
    return ldexpf(p, (int)ti);
#else
    return __expf(x);
#endif
}

#define BR_512   32
#define BC_512   32
#define HDIM_512 512
#define GLM_KVP_STRIDE 520
#define PAD_P_512 8
#define N_TILES_PER_WARP_512 16  // (HDIM/8) / 4 col-groups
#define TILE_CHUNKS_512 (BR_512 * (HDIM_512 / 8))

extern "C" __global__ void glm_sparse_mla_prefill_bf16_head32_tc_kv_pad(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    const int* __restrict__ token_indices,
    __nv_bfloat16* O,
    const unsigned int* __restrict__ block_table,
    unsigned int rows,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int index_width,
    unsigned int cache_block_size,
    float inv_sqrt_d
) {
    const unsigned int token_row = blockIdx.y;
    const unsigned int head_start = blockIdx.x * 32;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (token_row >= rows || head_start >= num_heads || head_dim != 512 || blockDim.x != 256) return;
    const unsigned int q_start = 0;
    const unsigned int q_len = min(32u, num_heads - head_start);
    const unsigned int q_tile_len = q_len;
    const unsigned int q_seq_stride = head_dim;
    const unsigned int kv_len = index_width;
    const int* indices = token_indices + (unsigned long long)token_row * index_width;
    Q += ((unsigned long long)token_row * num_heads + head_start) * head_dim;
    O += ((unsigned long long)token_row * num_heads + head_start) * head_dim;

    extern __shared__ __align__(16) unsigned char smem_dyn_512[];
    // 520 BF16 = 260 shared words: each QK row group advances four banks.
    // Both global caches and query tensors keep their original 512-value stride.
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_dyn_512);
    __nv_bfloat16* smem_K = smem_Q + (unsigned int)BR_512 * GLM_KVP_STRIDE;
    __nv_bfloat16* smem_P = smem_K + (unsigned int)BC_512 * GLM_KVP_STRIDE;
    float*         smem_ml = reinterpret_cast<float*>(
                       smem_P + (unsigned int)BR_512 * (BC_512 + PAD_P_512));

    if (tid < 64) smem_ml[tid] = (tid & 1) ? 0.0f : -1e30f;

    const unsigned int group_id     = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;
    const unsigned int qk_warp_m    = (warp_id & 1) * 16;          // warps 0-1
    const unsigned int pv_warp_m    = (warp_id & 1) * 16;          // pair: rows 0-15/16-31
    const unsigned int pv_n_start   = (warp_id >> 1) * N_TILES_PER_WARP_512;
    const unsigned int p_smem_stride = BC_512 + PAD_P_512;

    float acc_o[N_TILES_PER_WARP_512][4];
    #pragma unroll
    for (int i = 0; i < N_TILES_PER_WARP_512; i++) {
        acc_o[i][0]=0.f; acc_o[i][1]=0.f; acc_o[i][2]=0.f; acc_o[i][3]=0.f;
    }
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.f,    l_r1 = 0.f;

    unsigned int num_kv_blocks = (kv_len + BC_512 - 1) / BC_512;

    // === Initial Q + K[0] load (256 threads) ===
    {
        const unsigned int cpr = HDIM_512 / 8;
        for (unsigned int idx = tid; idx < TILE_CHUNKS_512; idx += blockDim.x) {
            unsigned int row = idx / cpr, col = (idx % cpr) * 8;
            unsigned int sa = __cvta_generic_to_shared(&smem_Q[row * GLM_KVP_STRIDE + col]);
            if (q_start + row < q_len) {
                const void* gm = (const void*)&Q[(q_start+row)*q_seq_stride + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gm));
            } else { *((uint4*)&smem_Q[row * GLM_KVP_STRIDE + col]) = make_uint4(0,0,0,0); }
        }
        if (num_kv_blocks > 0) {
            LOAD_KV_TILE_512(K_cache, block_table, smem_K, 0, kv_len, tid, blockDim.x);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * BC_512;
        unsigned int kv_end   = min(kv_start + BC_512, kv_len);
        unsigned int kv_tile_len = kv_end - kv_start;

        // QK and PV consume the same loaded latent tile. No second V load.
        float acc_s[4][4];
        if (warp_id < 2) {
            #pragma unroll
            for (int i=0;i<4;i++){acc_s[i][0]=0;acc_s[i][1]=0;acc_s[i][2]=0;acc_s[i][3]=0;}
            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K;

            #pragma unroll
            for (unsigned int ks = 0; ks < (HDIM_512/16); ks++) {
                unsigned int kb = ks*16;
                unsigned int ar0=qk_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=kb+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sQ[ar0*GLM_KVP_STRIDE+ac0];
                unsigned int a1=*(const unsigned int*)&sQ[ar1*GLM_KVP_STRIDE+ac0];
                unsigned int a2=*(const unsigned int*)&sQ[ar0*GLM_KVP_STRIDE+ac1];
                unsigned int a3=*(const unsigned int*)&sQ[ar1*GLM_KVP_STRIDE+ac1];
                #pragma unroll
                for (int nt=0; nt<4; nt++) {
                    unsigned int nc=nt*8+group_id, k0=kb+tid_in_group*2, k1=k0+8;
                    unsigned int b0=((unsigned int)sK[nc*GLM_KVP_STRIDE+k0+1]<<16)|(unsigned int)sK[nc*GLM_KVP_STRIDE+k0];
                    unsigned int b1=((unsigned int)sK[nc*GLM_KVP_STRIDE+k1+1]<<16)|(unsigned int)sK[nc*GLM_KVP_STRIDE+k1];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_s[nt][0]),"=f"(acc_s[nt][1]),"=f"(acc_s[nt][2]),"=f"(acc_s[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_s[nt][0]),"f"(acc_s[nt][1]),"f"(acc_s[nt][2]),"f"(acc_s[nt][3]));
                }
            }

            unsigned int row0=qk_warp_m+group_id, row1=row0+8;
            #pragma unroll
            for (int nt=0;nt<4;nt++) {
                acc_s[nt][0]*=inv_sqrt_d; acc_s[nt][1]*=inv_sqrt_d;
                acc_s[nt][2]*=inv_sqrt_d; acc_s[nt][3]*=inv_sqrt_d;
                unsigned int c0=nt*8+tid_in_group*2, c1=c0+1;
                const bool valid0 = c0 < kv_tile_len && indices[kv_start+c0] >= 0;
                const bool valid1 = c1 < kv_tile_len && indices[kv_start+c1] >= 0;
                if(!valid0){acc_s[nt][0]=-1e30f;acc_s[nt][2]=-1e30f;}
                if(!valid1){acc_s[nt][1]=-1e30f;acc_s[nt][3]=-1e30f;}
                if(row0>=q_tile_len){acc_s[nt][0]=-1e30f;acc_s[nt][1]=-1e30f;}
                if(row1>=q_tile_len){acc_s[nt][2]=-1e30f;acc_s[nt][3]=-1e30f;}

            }

            float rmax0=-1e30f, rmax1=-1e30f;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                rmax0=fmaxf(rmax0,fmaxf(acc_s[nt][0],acc_s[nt][1]));
                rmax1=fmaxf(rmax1,fmaxf(acc_s[nt][2],acc_s[nt][3]));
            }
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,1));
            rmax0=fmaxf(rmax0,__shfl_xor_sync(0xFFFFFFFF,rmax0,2));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,1));
            rmax1=fmaxf(rmax1,__shfl_xor_sync(0xFFFFFFFF,rmax1,2));

            float mn0=fmaxf(m_r0,rmax0);
            if (mn0 != m_r0) {
                float eo0=glm_kvp_exp(m_r0-mn0); l_r0*=eo0;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][0]*=eo0;acc_o[i][1]*=eo0;}
                m_r0=mn0;
            }
            float mn1=fmaxf(m_r1,rmax1);
            if (mn1 != m_r1) {
                float eo1=glm_kvp_exp(m_r1-mn1); l_r1*=eo1;
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][2]*=eo1;acc_o[i][3]*=eo1;}
                m_r1=mn1;
            }

            float sum0=0, sum1=0;
            #pragma unroll
            for(int nt=0;nt<4;nt++){
                const unsigned int item0 = nt*8+tid_in_group*2, item1 = item0+1;
                const bool valid0 = item0 < kv_tile_len && indices[kv_start+item0] >= 0;
                const bool valid1 = item1 < kv_tile_len && indices[kv_start+item1] >= 0;
                float p00=valid0 ? glm_kvp_exp(acc_s[nt][0]-m_r0) : 0.0f;
                float p01=valid1 ? glm_kvp_exp(acc_s[nt][1]-m_r0) : 0.0f;
                float p10=valid0 ? glm_kvp_exp(acc_s[nt][2]-m_r1) : 0.0f;
                float p11=valid1 ? glm_kvp_exp(acc_s[nt][3]-m_r1) : 0.0f;
                sum0+=p00+p01; sum1+=p10+p11;
                unsigned int c0=nt*8+tid_in_group*2;
                smem_P[row0*p_smem_stride+c0]   = __float2bfloat16(p00);
                smem_P[row0*p_smem_stride+c0+1] = __float2bfloat16(p01);
                smem_P[row1*p_smem_stride+c0]   = __float2bfloat16(p10);
                smem_P[row1*p_smem_stride+c0+1] = __float2bfloat16(p11);
            }
            sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,1); sum0+=__shfl_xor_sync(0xFFFFFFFF,sum0,2);
            sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,1); sum1+=__shfl_xor_sync(0xFFFFFFFF,sum1,2);
            l_r0+=sum0; l_r1+=sum1;

            if(tid_in_group==0){
                smem_ml[row0*2  ]=m_r0; smem_ml[row0*2+1]=l_r0;
                smem_ml[row1*2  ]=m_r1; smem_ml[row1*2+1]=l_r1;
            }
            asm volatile("cp.async.commit_group;");
        } else {
            // Keep the existing commit/wait structure with an empty group.
            asm volatile("cp.async.commit_group;");
        }

        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 2-7 rescale to current m before PV
        if(warp_id>=2){
            unsigned int r0=pv_warp_m+group_id, r1=r0+8;
            float cm0=smem_ml[r0*2], cm1=smem_ml[r1*2];
            if (cm0 != m_r0) {
                float er0=glm_kvp_exp(m_r0-cm0);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][0]*=er0;acc_o[i][1]*=er0;}
                m_r0=cm0;
            }
            if (cm1 != m_r1) {
                float er1=glm_kvp_exp(m_r1-cm1);
                #pragma unroll
                for(int i=0;i<N_TILES_PER_WARP_512;i++){acc_o[i][2]*=er1;acc_o[i][3]*=er1;}
                m_r1=cm1;
            }
        }

        // === PV MMA (all 8 warps) ===
        {
            const unsigned short* sP=(const unsigned short*)smem_P;
            const unsigned short* sV=(const unsigned short*)smem_K;
            #pragma unroll
            for(unsigned int ks=0;ks<2;ks++){
                unsigned int ko=ks*16;
                unsigned int ar0=pv_warp_m+group_id, ar1=ar0+8;
                unsigned int ac0=ko+tid_in_group*2, ac1=ac0+8;
                unsigned int a0=*(const unsigned int*)&sP[ar0*p_smem_stride+ac0];
                unsigned int a1=*(const unsigned int*)&sP[ar1*p_smem_stride+ac0];
                unsigned int a2=*(const unsigned int*)&sP[ar0*p_smem_stride+ac1];
                unsigned int a3=*(const unsigned int*)&sP[ar1*p_smem_stride+ac1];
                #pragma unroll
                for(int nt=0;nt<N_TILES_PER_WARP_512;nt++){
                    unsigned int nc=(pv_n_start+nt)*8+group_id, k0=ko+tid_in_group*2, k1=k0+8;
                    unsigned int b0=((unsigned int)sV[(k0+1)*GLM_KVP_STRIDE+nc]<<16)|(unsigned int)sV[k0*GLM_KVP_STRIDE+nc];
                    unsigned int b1=((unsigned int)sV[(k1+1)*GLM_KVP_STRIDE+nc]<<16)|(unsigned int)sV[k1*GLM_KVP_STRIDE+nc];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        :"=f"(acc_o[nt][0]),"=f"(acc_o[nt][1]),"=f"(acc_o[nt][2]),"=f"(acc_o[nt][3])
                        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                         "f"(acc_o[nt][0]),"f"(acc_o[nt][1]),"f"(acc_o[nt][2]),"f"(acc_o[nt][3]));
                }
            }
        }

        __syncthreads();

        // === Sequential K[next] load (single-buffered, all 256 threads) ===
        if(kv_block+1 < num_kv_blocks){
            LOAD_KV_TILE_512(K_cache, block_table, smem_K, (kv_block+1)*BC_512, kv_len, tid, blockDim.x);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncthreads();
        }
    }

    // === Final normalization and store ===
    {
        unsigned int r0=pv_warp_m+group_id, r1=r0+8;
        float il0,il1;
        if(warp_id<2){
            il0=(l_r0>0)?(1.f/l_r0):0;
            il1=(l_r1>0)?(1.f/l_r1):0;
        } else {
            float lv0=smem_ml[r0*2+1], lv1=smem_ml[r1*2+1];
            il0=(lv0>0)?(1.f/lv0):0;
            il1=(lv1>0)?(1.f/lv1):0;
        }

        __nv_bfloat16* ob = O;
        #pragma unroll
        for(int nt=0;nt<N_TILES_PER_WARP_512;nt++){
            unsigned int c0 = (pv_n_start+nt)*8 + tid_in_group*2;
            unsigned int gr0 = q_start+r0, gr1 = q_start+r1;
            if(gr0<q_len && r0<q_tile_len && c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0]*il0));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1]*il0));
                *(unsigned int*)&ob[gr0*q_seq_stride+c0]=lo|(hi<<16);
            }
            if(gr1<q_len && r1<q_tile_len && c0<head_dim){
                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2]*il1));
                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3]*il1));
                *(unsigned int*)&ob[gr1*q_seq_stride+c0]=lo|(hi<<16);
            }
        }
    }
}
