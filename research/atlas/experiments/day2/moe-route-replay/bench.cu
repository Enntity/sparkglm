// SPDX-License-Identifier: AGPL-3.0-only
#include "capture_routes.hpp"
#include "oracle.hpp"
#ifndef ATLAS_HOST_ONLY
#include "pipeline.cuh"
struct LayerTiming {double serial_ms,joint_ms;};
struct Case {
    Weights&w;const ReplayLayer&capture;std::vector<CaseRoutes>owners;CaseRoutes flat;
    std::unique_ptr<Pipeline>serial[4][2],joint[2];
    std::unique_ptr<Buffer<uint16_t>>serial_sum[4],scattered[4];
    Buffer<uint16_t>joint_sum{12*H};cudaStream_t stream;Clock clock;
    Case(Weights&w,const ReplayLayer&capture,cudaStream_t stream):w(w),capture(capture),owners(routes_from_capture(capture)),flat(owners),stream(stream){
        for(unsigned o=0;o<4;++o){for(unsigned r=0;r<2;++r)serial[o][r]=std::make_unique<Pipeline>(r,owners[o],w);
            serial_sum[o]=std::make_unique<Buffer<uint16_t>>(3*H);scattered[o]=std::make_unique<Buffer<uint16_t>>(3*H);}
        for(unsigned r=0;r<2;++r)joint[r]=std::make_unique<Pipeline>(r,flat,w);
        size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=4*GiB,"4GiB free reserve AFTER all case allocations");
        std::printf("MEMORY round=%u layer=%u live_bytes=%zu peak_bytes=%zu free_bytes=%zu\n",capture.round,capture.layer,live_bytes,peak_bytes,free);
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
    LayerTiming run(){
        verify(true);
        std::vector<double>samples[2];
        for(unsigned pair=0;pair<17;++pair){double critical[2]={0,0};
            for(unsigned ai=0;ai<2;++ai){unsigned arm=(pair+ai)&1;
                if(arm==0){for(unsigned o=0;o<4;++o){double ranks[2]={0,0};
                    for(unsigned ri=0;ri<2;++ri){unsigned rank=(pair+ri)&1;auto&v=*serial[o][rank];v.reset(stream);
                        ranks[rank]=clock.run(stream,[&]{v.launch(stream);});}
                    serial_sum[o]->poison(stream);double add=pair_add(*serial[o][0],*serial[o][1],*serial_sum[o],stream,clock);
                    critical[0]+=std::max(ranks[0],ranks[1])+add;
                    if(pair>=2)std::printf("SERIAL_OWNER round=%u layer=%u pair=%u owner=%u rank0_ms=%.9f rank1_ms=%.9f add_ms=%.9f\n",capture.round,capture.layer,pair-2,o,ranks[0],ranks[1],add);
                }}else{double ranks[2]={0,0};
                    for(unsigned ri=0;ri<2;++ri){unsigned rank=(pair+ri)&1;auto&v=*joint[rank];v.reset(stream);v.poison_joint_inputs(stream);
                        ranks[rank]=clock.run(stream,[&]{launch_joint(rank);});}
                    joint_sum.poison(stream);double add=pair_add(*joint[0],*joint[1],joint_sum,stream,clock);
                    for(auto&v:scattered)v->poison(stream);double copy=scatter();critical[1]=std::max(ranks[0],ranks[1])+add+copy;
                    if(pair>=2)std::printf("JOINT_PAIR round=%u layer=%u pair=%u rank0_pack_pipeline_ms=%.9f rank1_pack_pipeline_ms=%.9f add_ms=%.9f scatter_ms=%.9f\n",capture.round,capture.layer,pair-2,ranks[0],ranks[1],add,copy);
                }
            }
            if(pair>=2){samples[0].push_back(critical[0]);samples[1].push_back(critical[1]);
                std::printf("PAIR round=%u layer=%u pair=%u serial_critical_ms=%.9f joint_critical_ms=%.9f\n",capture.round,capture.layer,pair-2,critical[0],critical[1]);}
        }
        verify(false);double serial_ms=median(samples[0]),joint_ms=median(samples[1]),gain=serial_ms/joint_ms;
        std::printf("REPLAY_LAYER round=%u layer=%u serial_critical_ms=%.9f joint_inclusive_ms=%.9f speedup=%.9f network_included=false\n",capture.round,capture.layer,serial_ms,joint_ms,gain);
        return {serial_ms,joint_ms};
    }
};
#endif
int main(){host_tests();joint_host_tests();
    for(unsigned g=0;g<2;++g)for(unsigned l=0;l<42;++l){
        const auto&c=REPLAY_LAYERS[g][l];need(c.round==(g?8u:1u)&&c.layer==l+3,"fixed all-layer selection/order");
        auto r=routes_from_capture(c);CaseRoutes flat(r);check_routes(flat);
        for(unsigned rank=0;rank<2;++rank)expected_work(flat,rank);
    }
    std::printf("REPLAY_BINDING routes_sha256=%s analysis_sha256=%s rounds=1,8 layers=3..44 threshold=1.3\n",ROUTES_SHA256,ANALYSIS_SHA256);
#ifndef ATLAS_HOST_ONLY
    size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=12*GiB,"need8GiB cap plus4GiB reserve before allocations");
    std::printf("GPU before free_bytes=%zu total_bytes=%zu\n",free,total);Weights weights;
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(unsigned g=0;g<2;++g){double serial_sum=0,joint_sum=0;
        for(unsigned l=0;l<42;++l){const auto&capture=REPLAY_LAYERS[g][l];
            std::printf("REPLAY_LAYER_BEGIN round=%u layer=%u\n",capture.round,capture.layer);
            {Case test(weights,capture,stream);auto timing=test.run();serial_sum+=timing.serial_ms;joint_sum+=timing.joint_ms;}
        }
        weights.check();double gain=serial_sum/joint_sum;
        std::printf("REPLAY_AGGREGATE round=%u layers=42 serial_sum_ms=%.9f joint_sum_ms=%.9f speedup=%.9f threshold=1.3\n",g?8:1,serial_sum,joint_sum,gain);
        if(gain<1.3){CK(cudaStreamDestroy(stream));std::puts("REJECT fixed aggregate routed-pipeline gate; no replacement round or layer selection");return 3;}
    }
    CK(cudaStreamDestroy(stream));std::puts("PASS primary1 and confirmation8 aggregate gates; engine/C4 remains unqualified");
#endif
}
