// SPDX-License-Identifier: AGPL-3.0-only
#include "fixture.hpp"
#ifndef ATLAS_HOST_ONLY
#include "gpu_fixture.cuh"
#include "oracle.hpp"
#include "production/kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "production/kernels/gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu"
#include "production/kernels/gb10/common/quantize_bf16_to_nvfp4.cu"
#include "production/kernels/gb10/common/moe_permute.cu"
#include "production/kernels/gb10/common/bf16_add.cu"
using BF=__nv_bfloat16;
struct Pipeline {
    bool tp;unsigned rank,inter;const Routes&r;Weights&w;
    Buffer<uint16_t> input{M*H},gate,up,down{R*H},out{M*H};
    Buffer<unsigned> top{R},work;Buffer<float> routeweights{R};
    Buffer<int>tokens{R},experts{R},offsets{E+1},reverse{R},total{1};
    Buffer<unsigned char>a{M*H/2},aq{M*H/16},stage,packed;
    Pipeline(bool tp,unsigned rank,const Routes&r,Weights&w):tp(tp),rank(rank),inter(tp?I/2:I),r(r),w(w),gate(R*inter),up(R*inter),work(R*(inter/128)*2),stage(R*inter*9/16),packed(R*inter*9/16) {
        input.put(r.input.data());top.put(r.ids.data());routeweights.put(r.weights.data());
    }
    void reset(cudaStream_t s){gate.poison(s);up.poison(s);down.poison(s);out.poison(s);stage.poison(s);packed.poison(s);}
    void launch(cudaStream_t s){auto&g=*w.table[tp][rank][0];auto&u=*w.table[tp][rank][1];auto&d=*w.table[tp][rank][2];
        moe_sort_by_expert<<<1,256,0,s>>>(top.p,tokens.p,experts.p,offsets.p,reverse.p,R,E,TOP);CK(cudaGetLastError());
        moe_build_tile_worklist<<<1,256,0,s>>>(offsets.p,g.w.p,work.p,total.p,E,inter/128,64);CK(cudaGetLastError());
        quantize_bf16_to_nvfp4<<<M,128,0,s>>>(reinterpret_cast<const BF*>(input.p),a.p,aq.p,1.0f,M,H);CK(cudaGetLastError());
        moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up<<<dim3(R*(inter/128),2),128,0,s>>>(
            a.p,aq.p,g.w.p,g.s.p,g.s2.p,reinterpret_cast<BF*>(gate.p),u.w.p,u.s.p,u.s2.p,reinterpret_cast<BF*>(up.p),offsets.p,tokens.p,E,inter,H,work.p,total.p,R*(inter/128));CK(cudaGetLastError());
        silu_mul_quant_nvfp4<<<R,128,0,s>>>(reinterpret_cast<const BF*>(gate.p),reinterpret_cast<const BF*>(up.p),stage.p,stage.p+R*inter/2,nullptr,R,inter);CK(cudaGetLastError());
        CK(cudaMemcpyAsync(packed.p,stage.p,stage.n,cudaMemcpyDeviceToDevice,s));
        // Exact current production down dispatch: dense expert grid, not a new compact variant.
        moe_w4a4_grouped_gemm_prequant_t_k64_vecscale<<<dim3(H/128,1,E),128,0,s>>>(packed.p,packed.p+R*inter/2,d.w.p,d.s.p,d.s2.p,reinterpret_cast<BF*>(down.p),offsets.p,nullptr,E,H,inter);CK(cudaGetLastError());
        moe_unpermute_reduce_indexed_ep<<<M,256,0,s>>>(reinterpret_cast<const BF*>(down.p),reinterpret_cast<BF*>(out.p),reverse.p,reinterpret_cast<const int*>(top.p),routeweights.p,H,M,TOP,tp?0:rank*144,tp?E:(rank+1)*144);CK(cudaGetLastError());
    }
    Snapshot snap(){return Snapshot{inter,rank,tp,top.read(),routeweights.read(),tokens.read(),experts.read(),offsets.read(),reverse.read(),input.read(),gate.read(),up.read(),down.read(),out.read(),a.read(),aq.read(),packed.read()};}
    void guards(){input.guards();gate.guards();up.guards();down.guards();out.guards();top.guards();work.guards();routeweights.guards();tokens.guards();experts.guards();offsets.guards();reverse.guards();total.guards();a.guards();aq.guards();stage.guards();packed.guards();
        need(total.read()[0]>=0&&total.read()[0]<=int(R*(inter/128)),"worklist capacity");}
};
struct Clock {
    cudaEvent_t a,b;Clock(){CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));}~Clock(){cudaEventDestroy(a);cudaEventDestroy(b);}
    template<class F>double run(cudaStream_t s,F f){CK(cudaEventRecord(a,s));f();CK(cudaEventRecord(b,s));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));return ms;}
};
static double pair_add(Pipeline& a,Pipeline& b,Buffer<uint16_t>&sum,cudaStream_t stream,Clock&clock){
    // Copy is harness-only preservation of rank0. Production NCCL adds in-place.
    CK(cudaMemcpyAsync(sum.p,a.out.p,M*H*2,cudaMemcpyDeviceToDevice,stream));
    return clock.run(stream,[&]{bf16_add_inplace<<<(M*H+255)/256,256,0,stream>>>(reinterpret_cast<BF*>(sum.p),reinterpret_cast<const BF*>(b.out.p),M*H);CK(cudaGetLastError());});
}
static double run_case(Weights&w,unsigned owners,bool skew,cudaStream_t stream){
    std::vector<std::unique_ptr<Routes>>routes;std::unique_ptr<Pipeline>p[4][2][2];
    Buffer<uint16_t>sum{M*H};Clock clock;
    for(unsigned o=0;o<owners;++o){routes.emplace_back(std::make_unique<Routes>(skew,o));for(unsigned arm=0;arm<2;++arm)for(unsigned rank=0;rank<2;++rank)p[o][arm][rank]=std::make_unique<Pipeline>(arm,rank,*routes[o],w);}
    size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=4*GiB,"4GiB free reserve after all allocations");
    std::printf("MEMORY owners=%u skew=%u live_bytes=%zu peak_bytes=%zu free_bytes=%zu\n",owners,skew,live_bytes,peak_bytes,free);
    for(unsigned o=0;o<owners;++o){std::array<Snapshot,4>snaps;std::vector<uint16_t>sums[2];
        for(unsigned arm=0;arm<2;++arm){for(unsigned rank=0;rank<2;++rank){auto&v=*p[o][arm][rank];v.reset(stream);v.launch(stream);CK(cudaStreamSynchronize(stream));v.guards();snaps[arm*2+rank]=v.snap();}
            pair_add(*p[o][arm][0],*p[o][arm][1],sum,stream,clock);sums[arm]=sum.read();}
        compare(snaps,*routes[o],sums[0],sums[1]);sum.guards();
    }
    std::vector<double>aggregate[2];std::vector<double>rank_times[4][2][2],add_times[4][2];
    for(unsigned pair=0;pair<17;++pair){double critical[2]={0,0};
        for(unsigned o=0;o<owners;++o)for(unsigned ai=0;ai<2;++ai){unsigned arm=(pair+ai)&1;double ranks[2]={0,0};
            for(unsigned ri=0;ri<2;++ri){unsigned rank=(pair+ri)&1;auto&v=*p[o][arm][rank];v.reset(stream);
                ranks[rank]=clock.run(stream,[&]{v.launch(stream);});if(pair>=2)rank_times[o][arm][rank].push_back(ranks[rank]);}
            double add=pair_add(*p[o][arm][0],*p[o][arm][1],sum,stream,clock);
            critical[arm]+=std::max(ranks[0],ranks[1])+add;if(pair>=2)add_times[o][arm].push_back(add);
            if(pair>=2)std::printf("OWNER_PAIR owners=%u skew=%u pair=%u owner=%u tp=%u rank0_ms=%.9f rank1_ms=%.9f add_ms=%.9f\n",owners,skew,pair-2,o,arm,ranks[0],ranks[1],add);
        }
        if(pair>=2){aggregate[0].push_back(critical[0]);aggregate[1].push_back(critical[1]);
            std::printf("PAIR owners=%u skew=%u pair=%u ep_critical_ms=%.9f tp_critical_ms=%.9f\n",owners,skew,pair-2,critical[0],critical[1]);}
    }
    for(unsigned o=0;o<owners;++o)for(unsigned arm=0;arm<2;++arm){for(unsigned rank=0;rank<2;++rank){auto&v=*p[o][arm][rank];v.guards();check_snapshot(v.snap(),*routes[o]);
        std::printf("RANK owners=%u skew=%u owner=%u tp=%u rank=%u median_ms=%.9f\n",owners,skew,o,arm,rank,median(rank_times[o][arm][rank]));}
        std::printf("PAIR_ADD owner=%u tp=%u median_ms=%.9f\n",o,arm,median(add_times[o][arm]));}
    sum.guards();double ep=median(aggregate[0]),tp=median(aggregate[1]),gain=ep/tp;
    std::printf("RESULT {\"owners\":%u,\"skew\":%s,\"ep_critical_ms\":%.9f,\"tp_critical_ms\":%.9f,\"speedup\":%.9f,\"threshold\":1.3,\"network_included\":false}\n",owners,skew?"true":"false",ep,tp,gain);return gain;
}
#endif
int main(){host_tests();
#ifndef ATLAS_HOST_ONLY
    size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=12*GiB,"need8GiB cap plus4GiB free reserve before allocations");
    std::printf("GPU before free_bytes=%zu total_bytes=%zu\n",free,total);Weights weights;
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    const std::array<std::pair<unsigned,bool>,4> cases={{{1,false},{1,true},{4,false},{4,true}}};
    for(auto c:cases){double gain=run_case(weights,c.first,c.second,stream);weights.check();
        if(gain<1.3){std::puts("REJECT fixed1.3 critical-path gate; remaining fixtures skipped");CK(cudaStreamDestroy(stream));return 3;}}
    CK(cudaStreamDestroy(stream));std::puts("PASS all four standalone gates; model/topology integration remains unqualified");
#endif
}
