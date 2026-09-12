// SPDX-License-Identifier: AGPL-3.0-only
// GLM N288/K4096 prefill specialization. The BM16/BN32 body is the standalone
// oracle winner: one accumulator per output, k0..K-1, separate FMUL/FADD.
// Grid (ceil(N/32),ceil(M/16),1), block (8,16,1); no model/cache allocation.
// sB[][65] has no different-address bank collision across the eight x lanes.
#pragma once

extern "C" __global__ void dense_gemm_bf16_router_bn32(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned M,unsigned N,unsigned K
) {
    constexpr unsigned BM=16, BN=32, BK=64, THREADS=8*BM;
    __shared__ float sA[BM][BK+1];
    __shared__ float sB[BN][BK+1];
    const unsigned tid=threadIdx.y*8+threadIdx.x;
    const unsigned row0=blockIdx.y*BM, col0=blockIdx.x*BN;
    const unsigned row=row0+threadIdx.y, col=col0+threadIdx.x*4;
    float acc[4]={0.f,0.f,0.f,0.f};
    for(unsigned kb=0;kb<K;kb+=BK) {
        // Both variants use aligned uint4 loads, including tail-row padding.
        for(unsigned e=tid*8;e<BM*BK;e+=THREADS*8) {
            unsigned r=e/BK,c=e%BK;
            if(row0+r<M && kb+c+7<K) {
                uint4 v=*reinterpret_cast<const uint4*>(A+size_t(row0+r)*K+kb+c);
                const unsigned short* b=reinterpret_cast<const unsigned short*>(&v);
                #pragma unroll
                for(unsigned j=0;j<8;j++) sA[r][c+j]=__bfloat162float(__ushort_as_bfloat16(b[j]));
            } else {
                #pragma unroll
                for(unsigned j=0;j<8;j++) sA[r][c+j]=(row0+r<M && kb+c+j<K)
                    ?__bfloat162float(A[size_t(row0+r)*K+kb+c+j]):0.f;
            }
        }
        for(unsigned e=tid*8;e<BN*BK;e+=THREADS*8) {
            unsigned r=e/BK,c=e%BK;
            if(col0+r<N && kb+c+7<K) {
                uint4 v=*reinterpret_cast<const uint4*>(B+size_t(col0+r)*K+kb+c);
                const unsigned short* b=reinterpret_cast<const unsigned short*>(&v);
                #pragma unroll
                for(unsigned j=0;j<8;j++) sB[r][c+j]=__bfloat162float(__ushort_as_bfloat16(b[j]));
            } else {
                #pragma unroll
                for(unsigned j=0;j<8;j++) sB[r][c+j]=(col0+r<N && kb+c+j<K)
                    ?__bfloat162float(B[size_t(col0+r)*K+kb+c+j]):0.f;
            }
        }
        __syncthreads();
        #pragma unroll 8
        for(unsigned kk=0;kk<BK;kk++) {
            float a=sA[threadIdx.y][kk];
            #pragma unroll
            for(unsigned j=0;j<4;j++)
                acc[j]=__fadd_rn(acc[j],__fmul_rn(a,sB[threadIdx.x*4+j][kk]));
        }
        __syncthreads();
    }
    if(row<M) {
        #pragma unroll
        for(unsigned j=0;j<4;j++) if(col+j<N) C[size_t(row)*N+col+j]=__float2bfloat16_rn(acc[j]);
    }
}
