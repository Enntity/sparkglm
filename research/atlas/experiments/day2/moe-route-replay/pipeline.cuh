// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "gpu_fixture.cuh"
#include "owner_cohort_copy.cu"
#include "oracle.hpp"
#include "production/kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "production/kernels/gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu"
#include "production/kernels/gb10/common/quantize_bf16_to_nvfp4.cu"
#include "production/kernels/gb10/common/moe_permute.cu"
#include "production/kernels/gb10/common/bf16_add.cu"
using BF=__nv_bfloat16;
struct Pipeline {
    unsigned rank,rows,expanded;const CaseRoutes&r;Weights&w;
    Buffer<uint16_t>input,gate,up,down,out;
    Buffer<unsigned>top,work;Buffer<float>routeweights;
    Buffer<int>tokens,experts,offsets{E+1},reverse;
    Buffer<unsigned char>a,aq,stage,packed;
    Pipeline(unsigned rank,const CaseRoutes&r,Weights&w):rank(rank),rows(checked_rows(r,rank)),expanded(rows*TOP),r(r),w(w),input(rows*H),gate(expanded*I),up(expanded*I),down(expanded*H),out(rows*H),top(expanded),work(work_bytes(rows)/4),routeweights(expanded),tokens(expanded),experts(expanded),reverse(expanded),a(rows*H/2),aq(rows*H/16),stage(expanded*I*9/16),packed(expanded*I*9/16){
        check_routes(r);need(work_admitted(rows,work.n*4),"work capacity before builder");need(rank<2,"rank bound");
        input.put(r.input.data());top.put(r.ids.data());routeweights.put(r.weights.data());
    }
    void reset(cudaStream_t s){gate.poison(s);up.poison(s);down.poison(s);out.poison(s);work.poison(s);stage.poison(s);packed.poison(s);}
    void poison_joint_inputs(cudaStream_t s){need(rows==12,"joint pack rows");input.poison(s);top.poison(s);routeweights.poison(s);}
    void pack(const std::array<Pipeline*,4>&owners,cudaStream_t s){
        need(rows==12,"joint pack row extent");
        for(unsigned o=0;o<4;++o)need(owners[o]->rows==3&&owners[o]->rank==rank,"pack owner geometry");
        // Qualified exact-byte utility; all three gathers are INSIDE pipeline events.
        CK(static_cast<cudaError_t>(copy_four_segments(owners[0]->input.p,owners[1]->input.p,owners[2]->input.p,owners[3]->input.p,input.p,3*H*2,0,reinterpret_cast<void*>(s))));
        CK(static_cast<cudaError_t>(copy_four_segments(owners[0]->top.p,owners[1]->top.p,owners[2]->top.p,owners[3]->top.p,top.p,24*4,0,reinterpret_cast<void*>(s))));
        CK(static_cast<cudaError_t>(copy_four_segments(owners[0]->routeweights.p,owners[1]->routeweights.p,owners[2]->routeweights.p,owners[3]->routeweights.p,routeweights.p,24*4,0,reinterpret_cast<void*>(s))));
    }
    void launch(cudaStream_t s){auto&g=*w.table[0][rank][0];auto&u=*w.table[0][rank][1];auto&d=*w.table[0][rank][2];
        moe_sort_by_expert<<<1,256,0,s>>>(top.p,tokens.p,experts.p,offsets.p,reverse.p,expanded,E,TOP);CK(cudaGetLastError());
        moe_build_tile_worklist<<<1,256,0,s>>>(offsets.p,g.w.p,work.p+4,reinterpret_cast<int*>(work.p),E,I/128,64);CK(cudaGetLastError());
        quantize_bf16_to_nvfp4<<<rows,128,0,s>>>(reinterpret_cast<const BF*>(input.p),a.p,aq.p,1.0f,rows,H);CK(cudaGetLastError());
        moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up<<<dim3(expanded*(I/128),2),128,0,s>>>(a.p,aq.p,g.w.p,g.s.p,g.s2.p,reinterpret_cast<BF*>(gate.p),u.w.p,u.s.p,u.s2.p,reinterpret_cast<BF*>(up.p),offsets.p,tokens.p,E,I,H,work.p+4,reinterpret_cast<int*>(work.p),expanded*(I/128));CK(cudaGetLastError());
        silu_mul_quant_nvfp4<<<expanded,128,0,s>>>(reinterpret_cast<const BF*>(gate.p),reinterpret_cast<const BF*>(up.p),stage.p,stage.p+expanded*I/2,nullptr,expanded,I);CK(cudaGetLastError());
        CK(cudaMemcpyAsync(packed.p,stage.p,stage.n,cudaMemcpyDeviceToDevice,s));
        // Unique top8 per token proves M_expert<=rows<=12<64: one dense M tile.
        moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<dim3(H/128,1,E),128,0,s>>>(packed.p,packed.p+expanded*I/2,d.w.p,d.s.p,d.s2.p,reinterpret_cast<BF*>(down.p),offsets.p,nullptr,E,H,I);CK(cudaGetLastError());
        moe_unpermute_reduce_indexed_ep<<<rows,256,0,s>>>(reinterpret_cast<const BF*>(down.p),reinterpret_cast<BF*>(out.p),reverse.p,reinterpret_cast<const int*>(top.p),routeweights.p,H,rows,TOP,rank*144,(rank+1)*144);CK(cudaGetLastError());
    }
    Snapshot snap(){return Snapshot{rows,I,rank,false,top.read(),routeweights.read(),tokens.read(),experts.read(),offsets.read(),reverse.read(),input.read(),gate.read(),up.read(),down.read(),out.read(),a.read(),aq.read(),packed.read()};}
    void guards(){input.guards();gate.guards();up.guards();down.guards();out.guards();top.guards();work.guards();routeweights.guards();tokens.guards();experts.guards();offsets.guards();reverse.guards();a.guards();aq.guards();stage.guards();packed.guards();check_work(work.read(),r,rank);}
};
struct Clock {
    cudaEvent_t a,b;Clock(){CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));}~Clock(){cudaEventDestroy(a);cudaEventDestroy(b);}
    template<class F>double run(cudaStream_t s,F f){CK(cudaEventRecord(a,s));f();CK(cudaEventRecord(b,s));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));return ms;}
};
inline double pair_add(Pipeline&a,Pipeline&b,Buffer<uint16_t>&sum,cudaStream_t stream,Clock&clock){
    need(a.rows==b.rows&&sum.n==size_t(a.rows)*H,"pair extent");
    // Harness-only preservation outside event, excluded equally on both arms.
    CK(cudaMemcpyAsync(sum.p,a.out.p,sum.n*2,cudaMemcpyDeviceToDevice,stream));
    return clock.run(stream,[&]{bf16_add_inplace<<<(sum.n+255)/256,256,0,stream>>>(reinterpret_cast<BF*>(sum.p),reinterpret_cast<const BF*>(b.out.p),sum.n);CK(cudaGetLastError());});
}
