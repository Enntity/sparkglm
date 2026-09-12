// SPDX-License-Identifier: AGPL-3.0-only
#include "joint_fixture.hpp"
#include "oracle.hpp"
#ifndef ATLAS_HOST_ONLY
#include "pipeline.cuh"
struct Case {
    Weights&w;bool skew,shared;std::vector<CaseRoutes>owners;CaseRoutes flat;
    std::unique_ptr<Pipeline>serial[4][2],joint[2];
    std::unique_ptr<Buffer<uint16_t>>serial_sum[4],scattered[4];
    Buffer<uint16_t>joint_sum{12*H};cudaStream_t stream;Clock clock;
    Case(Weights&w,bool skew,bool shared,cudaStream_t stream):w(w),skew(skew),shared(shared),owners(owner_routes(skew,shared)),flat(owners),stream(stream){
        for(unsigned o=0;o<4;++o){for(unsigned r=0;r<2;++r)serial[o][r]=std::make_unique<Pipeline>(r,owners[o],w);
            serial_sum[o]=std::make_unique<Buffer<uint16_t>>(3*H);scattered[o]=std::make_unique<Buffer<uint16_t>>(3*H);}
        for(unsigned r=0;r<2;++r)joint[r]=std::make_unique<Pipeline>(r,flat,w);
        size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=4*GiB,"4GiB free reserve AFTER all case allocations");
        std::printf("MEMORY skew=%u correctness_shared=%u live_bytes=%zu peak_bytes=%zu free_bytes=%zu\n",skew,shared,live_bytes,peak_bytes,free);
    }
    std::array<Pipeline*,4>sources(unsigned rank){return {serial[0][rank].get(),serial[1][rank].get(),serial[2][rank].get(),serial[3][rank].get()};}
    void launch_joint(unsigned rank){joint[rank]->pack(sources(rank),stream);joint[rank]->launch(stream);}
    double scatter(){
        return clock.run(stream,[&]{CK(static_cast<cudaError_t>(copy_four_segments(scattered[0]->p,scattered[1]->p,scattered[2]->p,scattered[3]->p,joint_sum.p,3*H*2,1,reinterpret_cast<void*>(stream))));});
    }
    void verify(bool launch){
        std::array<std::array<Snapshot,2>,4>s;std::array<Snapshot,2>j;
        std::array<std::vector<uint16_t>,4>ss,js;
        for(unsigned o=0;o<4;++o){for(unsigned r=0;r<2;++r){auto&v=*serial[o][r];
                if(launch){v.reset(stream);v.launch(stream);CK(cudaStreamSynchronize(stream));}v.guards();s[o][r]=v.snap();}
            if(launch)pair_add(*serial[o][0],*serial[o][1],*serial_sum[o],stream,clock);ss[o]=serial_sum[o]->read();serial_sum[o]->guards();}
        for(unsigned r=0;r<2;++r){auto&v=*joint[r];if(launch){v.reset(stream);v.poison_joint_inputs(stream);launch_joint(r);CK(cudaStreamSynchronize(stream));}v.guards();j[r]=v.snap();}
        if(launch){pair_add(*joint[0],*joint[1],joint_sum,stream,clock);
            for(auto&v:scattered)v->poison(stream);scatter();}
        for(unsigned o=0;o<4;++o){js[o]=scattered[o]->read();scattered[o]->guards();}joint_sum.guards();
        auto packed_sum=joint_sum.read();
        for(unsigned o=0;o<4;++o)for(unsigned i=0;i<3*H;++i)need(packed_sum[size_t(o)*3*H+i]==js[o][i],"actual last pair/scatter outputs identical");
        compare_joint(s,j,owners,flat,ss,js);
    }
    double run(){
        verify(true);
        if(shared){std::puts("PASS max12 shared-expert CORRECTNESS ONLY; no timing/promotion metric");return 1.;}
        std::vector<double>samples[2];
        for(unsigned pair=0;pair<17;++pair){double critical[2]={0,0};
            for(unsigned ai=0;ai<2;++ai){unsigned arm=(pair+ai)&1;
                if(arm==0){for(unsigned o=0;o<4;++o){double ranks[2]={0,0};
                    for(unsigned ri=0;ri<2;++ri){unsigned rank=(pair+ri)&1;auto&v=*serial[o][rank];v.reset(stream);
                        ranks[rank]=clock.run(stream,[&]{v.launch(stream);});}
                    serial_sum[o]->poison(stream);double add=pair_add(*serial[o][0],*serial[o][1],*serial_sum[o],stream,clock);
                    critical[0]+=std::max(ranks[0],ranks[1])+add;
                    if(pair>=2)std::printf("SERIAL_OWNER skew=%u pair=%u owner=%u rank0_ms=%.9f rank1_ms=%.9f add_ms=%.9f\n",skew,pair-2,o,ranks[0],ranks[1],add);
                }}else{double ranks[2]={0,0};
                    for(unsigned ri=0;ri<2;++ri){unsigned rank=(pair+ri)&1;auto&v=*joint[rank];v.reset(stream);v.poison_joint_inputs(stream);
                        ranks[rank]=clock.run(stream,[&]{launch_joint(rank);});}
                    joint_sum.poison(stream);double add=pair_add(*joint[0],*joint[1],joint_sum,stream,clock);
                    for(auto&v:scattered)v->poison(stream);double copy=scatter();critical[1]=std::max(ranks[0],ranks[1])+add+copy;
                    if(pair>=2)std::printf("JOINT_PAIR skew=%u pair=%u rank0_pack_pipeline_ms=%.9f rank1_pack_pipeline_ms=%.9f add_ms=%.9f scatter_ms=%.9f\n",skew,pair-2,ranks[0],ranks[1],add,copy);
                }
            }
            if(pair>=2){samples[0].push_back(critical[0]);samples[1].push_back(critical[1]);
                std::printf("PAIR skew=%u pair=%u serial_critical_ms=%.9f joint_critical_ms=%.9f\n",skew,pair-2,critical[0],critical[1]);}
        }
        verify(false);double serial_ms=median(samples[0]),joint_ms=median(samples[1]),gain=serial_ms/joint_ms;
        std::printf("RESULT {\"skew\":%s,\"owners\":4,\"serial_critical_ms\":%.9f,\"joint_inclusive_ms\":%.9f,\"speedup\":%.9f,\"threshold\":1.3,\"cross_owner_expert_overlap\":false,\"network_included\":false,\"copy_recipe\":\"3_gathers_per_rank_plus1_scatter_qualified625a9a16\"}\n",skew?"true":"false",serial_ms,joint_ms,gain);
        return gain;
    }
};
#endif
int main(){host_tests();joint_host_tests();
#ifndef ATLAS_HOST_ONLY
    size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=12*GiB,"need8GiB cap plus4GiB reserve before allocations");
    std::printf("GPU before free_bytes=%zu total_bytes=%zu\n",free,total);Weights weights;
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    {Case shared(weights,false,true,stream);shared.run();}weights.check();
    bool all_pass=true;
    for(bool skew:{false,true}){double gain;{Case test(weights,skew,false,stream);gain=test.run();}weights.check();
        if(gain<1.3){all_pass=false;std::puts("CASE_FAIL fixed1.3 routed-expert pipeline gate; retaining other predeclared independent fixture");}}
    CK(cudaStreamDestroy(stream));
    if(!all_pass){std::puts("REJECT all-cases gate: at least one fixed fixture failed; no engine admission from partial pass");return 3;}
    std::puts("PASS both fixed routed-expert pipeline gates; engine/cohort remains unqualified");
#endif
}
