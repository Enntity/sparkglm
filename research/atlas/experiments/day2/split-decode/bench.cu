// SPDX-License-Identifier: AGPL-3.0-only
#include <cstdio>
#include "guarded.cuh"
#include "lse_oracle.hpp"
#include "production/kernels/gb10/deepseek-v4-flash/nvfp4/glm_sparse_prefill_kv_reuse.cu"
#include "generated.cuh"
#include "merge.cu"
#include "merge_checks.cuh"
#include "compare.cuh"
constexpr unsigned SHARED=69376;
constexpr size_t RESERVE=size_t(4)<<30;
struct Inputs {
    const Fixture&f;Guarded query,kv,ids,table,baseline,candidate,part_o,part_lse,lse;
    explicit Inputs(const Fixture&source):f(source),query(f.query.size()*2),kv(f.kv.size()*2),ids(f.ids.size()*4),table(f.table.size()*4),baseline(f.elements()*2),candidate(f.elements()*2),part_o(16*f.elements()*4),part_lse(16*f.spec.rows*HEADS*4),lse(f.spec.rows*HEADS*4){
        validate(f);query.upload(f.query.data());kv.upload(f.kv.data());ids.upload(f.ids.data());table.upload(f.table.data());
        size_t free,total;CUDA(cudaMemGetInfo(&free,&total));require(free>=RESERVE,"4GiB actual free reserve after allocations");
    }
    void unchanged(){query.unchanged(f.query.data());kv.unchanged(f.kv.data());ids.unchanged(f.ids.data());table.unchanged(f.table.data());}
    void poison(){baseline.poison();candidate.poison();part_o.poison();part_lse.poison();lse.poison();}
    void launch(unsigned splits,bool serial){unsigned calls=serial?f.spec.rows:1,rows=serial?1:f.spec.rows;
        for(unsigned call=0;call<calls;++call){size_t row=serial?call:0;
            auto*q=query.as<__nv_bfloat16>()+row*HEADS*DIM;auto*k=kv.as<__nv_bfloat16>();auto*selected=ids.as<int>()+row*f.spec.width;
            auto*out=(splits?candidate:baseline).as<__nv_bfloat16>()+row*HEADS*DIM;
            if(!splits)glm_sparse_mla_prefill_bf16_head32_tc_kv_pad<<<dim3(1,rows),256,SHARED>>>(q,k,k,selected,out,table.as<unsigned>(),rows,HEADS,DIM,f.spec.width,BLOCK,SCALE);
            else{auto*po=part_o.as<float>()+row*16*HEADS*DIM;auto*pl=part_lse.as<float>()+row*16*HEADS;
                atlas_sparse_decode_split<<<dim3(1,rows,splits),256,SHARED>>>(q,k,k,selected,po,table.as<unsigned>(),rows,HEADS,DIM,f.spec.width,BLOCK,SCALE,pl,splits);CUDA(cudaGetLastError());
                CUDA(static_cast<cudaError_t>(merge_attention_parts(po,pl,out,lse.as<float>()+row*HEADS,rows,HEADS,DIM,splits,nullptr)));}
            CUDA(cudaGetLastError());
        }
    }
    void partial_checks(unsigned splits,bool serial,const std::vector<double>&scores){
        auto po=part_o.floats(),pl=part_lse.floats(),merged=lse.floats();
        for(unsigned r=0;r<f.spec.rows;++r)for(unsigned h=0;h<HEADS;++h){size_t rh=size_t(r)*HEADS+h;
            check_lse(merged[rh],logsum(scores,rh*f.spec.width,f.spec.width));
            for(unsigned s=0;s<16;++s){size_t pi=serial?(size_t(r)*16+s)*HEADS+h:(size_t(s)*f.spec.rows+r)*HEADS+h;
                if(s>=splits){uint32_t raw;std::memcpy(&raw,&pl[pi],4);require(raw==0xffffffffu,"inactive LSE slot overwritten");
                    for(unsigned d=0;d<DIM;++d){std::memcpy(&raw,&po[pi*DIM+d],4);require(raw==0xffffffffu,"inactive output slot overwritten");}continue;}
                auto[lo,hi]=interval(f.spec.width,s,splits);double ref=logsum(scores,rh*f.spec.width+lo,hi-lo);check_lse(pl[pi],ref);
                for(unsigned d=0;d<DIM;++d){float v=po[pi*DIM+d];require(std::isfinite(v),"partial output finite/written");if(ref==-INFINITY)require(v==0,"empty partial output zero");}
            }
        }
    }
};
static void correctness(const Spec&s){Fixture f=make_fixture(s);auto ref=oracle(f),scores=reference_scores(f);Inputs in(f);
    for(unsigned serial=0;serial<(s.rows==3?2u:1u);++serial)for(unsigned splits:{1u,2u,4u,8u,16u}){
        in.poison();in.launch(0,serial);in.launch(splits,serial);CUDA(cudaDeviceSynchronize());
        auto a=in.baseline.output(),b=in.candidate.output();auto ae=compare(f,a,ref),be=compare(f,b,ref);
        if(splits==1)require(a==b,"S1 must exactly reproduce baseline BF16 bits");
        in.partial_checks(splits,serial,scores);in.unchanged();
        std::printf("{\"stage\":\"correctness\",\"case\":\"%s\",\"splits\":%u,\"serial\":%s,\"baseline_maxabs\":%.9g,\"candidate_maxabs\":%.9g,\"candidate_relL2\":%.9g,\"lse_abs_limit\":0.0002,\"pass\":true}\n",s.name,splits,serial?"true":"false",ae.maximum,be.maximum,be.relative);std::fflush(stdout);
    }
}
static bool timing(const Spec&s,bool serial,unsigned splits,bool primary){Fixture f=make_fixture(s);Inputs in(f);in.poison();
    for(unsigned repeat=0;repeat<20;++repeat){in.launch((repeat&1)?splits:0,serial);in.launch((repeat&1)?0:splits,serial);}CUDA(cudaDeviceSynchronize());
    cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));std::vector<float>times[2];
    for(unsigned pair=0;pair<9;++pair)for(unsigned a=0;a<2;++a){unsigned arm=(pair+a)&1;
        CUDA(cudaEventRecord(start));in.launch(arm?splits:0,serial);CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,start,end));
        require(std::isfinite(ms)&&ms>0,"positive event timing");times[arm].push_back(ms);}
    CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));
    auto ref=oracle(f),scores=reference_scores(f);compare(f,in.baseline.output(),ref);compare(f,in.candidate.output(),ref);in.partial_checks(splits,serial,scores);in.unchanged();
    auto a=times[0],b=times[1];std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());double gain=double(a[4])/b[4];
    std::printf("{\"stage\":\"timing\",\"width\":%u,\"rows\":%u,\"serial\":%s,\"splits\":%u,\"primary\":%s,\"required_speedup\":2.0,\"baseline_ms\":%.9g,\"split_merge_ms\":%.9g,\"speedup\":%.9g,\"warmup_each\":20,\"pairs\":9,\"baseline_samples\":",s.width,s.rows,serial?"true":"false",splits,primary?"true":"false",a[4],b[4],gain);
    samples(times[0]);std::printf(",\"split_merge_samples\":");samples(times[1]);std::printf(",\"pass\":%s}\n",(!primary||gain>=2)?"true":"false");std::fflush(stdout);return !primary||gain>=2;
}
static void resources(){cudaDeviceProp p;CUDA(cudaGetDeviceProperties(&p,0));require(p.major==12&&p.minor==1,"GB10 sm121 required");
    size_t free,total;CUDA(cudaMemGetInfo(&free,&total));require(free>=RESERVE+LIMIT,"128MiB cap plus4GiB free before allocations");
    CUDA(cudaFuncSetAttribute(glm_sparse_mla_prefill_bf16_head32_tc_kv_pad,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
    CUDA(cudaFuncSetAttribute(atlas_sparse_decode_split,cudaFuncAttributeMaxDynamicSharedMemorySize,SHARED));
    cudaFuncAttributes a,b;CUDA(cudaFuncGetAttributes(&a,glm_sparse_mla_prefill_bf16_head32_tc_kv_pad));CUDA(cudaFuncGetAttributes(&b,atlas_sparse_decode_split));
    require(a.maxDynamicSharedSizeBytes>=int(SHARED)&&b.maxDynamicSharedSizeBytes>=int(SHARED),"shared opt-in effective");
    std::printf("{\"stage\":\"resources\",\"baseline_regs\":%d,\"split_regs\":%d,\"dynamic_shared\":%u,\"initial_free_bytes\":%zu}\n",a.numRegs,b.numRegs,SHARED,free);
}
int main(int argc,char**argv){try{
    require(argc==2&&(std::strcmp(argv[1],"--check")==0||std::strcmp(argv[1],"--run")==0),"usage bench --check|--run");host_tests();resources();merge_checks();
    for(const auto&s:correctness_specs())correctness(s);std::puts("{\"stage\":\"correctness_complete\",\"fixtures\":14,\"splits_tested\":[1,2,4,8,16],\"pass\":true}");
    if(std::strcmp(argv[1],"--run")==0){for(unsigned width:{2051u,2048u,2049u,2050u})if(!timing({"primary",1,width,16384,Kind::Random,width-2048,false},false,8,true))return 3;
        Spec m3={"serial_m3",3,2051,16384,Kind::Random,3,false};if(!timing(m3,true,8,true))return 3;
        for(unsigned splits:{4u,16u}){timing({"exploratory",1,2051,16384,Kind::Random,3,false},false,splits,false);timing(m3,true,splits,false);}}
    require(live_bytes==0,"allocation leak");std::printf("{\"stage\":\"complete\",\"peak_device_bytes\":%zu,\"pass\":true}\n",peak_bytes);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 2;}}
