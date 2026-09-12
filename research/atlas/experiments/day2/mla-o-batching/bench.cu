// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "fixture.hpp"
#include "production/kernels/gb10/common/dense_gemm_tc.cu"
#include "production/kernels/gb10/common/dense_gemv_bf16.cu"
#include "production/kernels/gb10/common/dense_gemv_bf16_batchm.cu"
#define CUDA(x) do {auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)
constexpr size_t GUARD=256,CAP=256ull*1024*1024,RESERVE=4ull*1024*1024*1024;
static size_t live=0,peak=0;
__global__ void fill(uint16_t* a,uint16_t* w,unsigned n,unsigned h,bool tiny) {
    const uint64_t start=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,step=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=start;i<uint64_t(n)*h;i+=step)w[i]=w_bits(i,h,tiny);
    for(uint64_t i=start;i<uint64_t(M)*h;i+=step)a[i]=a_bits(unsigned(i/h),unsigned(i%h),tiny);
}
__global__ void validate_inputs(const uint16_t* a,const uint16_t* w,unsigned n,unsigned h,bool tiny,unsigned* errors) {
    const uint64_t start=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,step=uint64_t(gridDim.x)*blockDim.x;
    unsigned bad=0;
    for(uint64_t i=start;i<uint64_t(n)*h;i+=step)bad+=w[i]!=w_bits(i,h,tiny);
    for(uint64_t i=start;i<uint64_t(M)*h;i+=step)bad+=a[i]!=a_bits(unsigned(i/h),unsigned(i%h),tiny);
    if(bad)atomicAdd(errors,bad);
}
struct Guarded {
    unsigned char* base=nullptr;size_t bytes,total;
    explicit Guarded(size_t size):bytes(size),total(size+2*GUARD) {
        require(total<CAP && live<=CAP-total,"device allocation budget");
        CUDA(cudaMalloc(reinterpret_cast<void**>(&base),total));live+=total;peak=std::max(peak,live);
        CUDA(cudaMemset(base,0xa5,total));
    }
    ~Guarded(){if(base){cudaFree(base);live-=total;}}
    Guarded(const Guarded&)=delete;
    template<class T>T* ptr()const{return reinterpret_cast<T*>(base+GUARD);}
    void poison(){CUDA(cudaMemset(base+GUARD,0xff,bytes));}
    void guards()const{
        std::array<unsigned char,2*GUARD> host;
        CUDA(cudaMemcpy(host.data(),base,GUARD,cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(host.data()+GUARD,base+GUARD+bytes,GUARD,cudaMemcpyDeviceToHost));
        for(auto x:host)require(x==0xa5,"redzone modified");
    }
    std::vector<uint16_t> read()const{
        guards();std::vector<uint16_t> host(bytes/2);
        CUDA(cudaMemcpy(host.data(),base+GUARD,bytes,cudaMemcpyDeviceToHost));return host;
    }
};
struct Fixture {
    unsigned n,h,stride;bool tiny;
    Guarded a,w,error;
    std::array<std::unique_ptr<Guarded>,MODES> out;
    Fixture(unsigned n_,unsigned h_,bool tiny_):n(n_),h(h_),stride(n+16),tiny(tiny_),
        a(size_t(M)*h*2),w(size_t(n)*h*2),error(4){
        require(n%4==0 && h%8==0,"GEMV barrier/vector geometry");
        for(auto& o:out)o=std::make_unique<Guarded>(size_t(M)*stride*2);
        fill<<<4096,256>>>(a.ptr<uint16_t>(),w.ptr<uint16_t>(),n,h,tiny);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    }
    void immutable(){
        CUDA(cudaMemset(error.ptr<unsigned>(),0,4));
        validate_inputs<<<4096,256>>>(a.ptr<uint16_t>(),w.ptr<uint16_t>(),n,h,tiny,error.ptr<unsigned>());
        CUDA(cudaGetLastError());unsigned errors=1;CUDA(cudaMemcpy(&errors,error.ptr<unsigned>(),4,cudaMemcpyDeviceToHost));
        require(errors==0,"input payload modified");a.guards();w.guards();error.guards();
    }
    unsigned ld(unsigned mode,bool padded)const{return padded&&mode<2?stride:n;}
    void launch(unsigned mode,bool padded=false){
        require(mode<MODES,"launch mode");
        auto A=a.ptr<__nv_bfloat16>();auto W=w.ptr<__nv_bfloat16>();auto O=out[mode]->ptr<__nv_bfloat16>();
        const unsigned out_stride=ld(mode,padded);
        if(mode==0)for(unsigned r=0;r<M;++r){
            dense_gemv_bf16<<<(n+3)/4,256>>>(A+size_t(r)*h,W,O+size_t(r)*out_stride,n,h);CUDA(cudaGetLastError());
        }else if(mode==1)for(unsigned owner=0;owner<4;++owner){
            const unsigned r=owner*3;
            dense_gemv_bf16_batchm<<<(n+3)/4,256>>>(A+size_t(r)*h,W,O+size_t(r)*out_stride,3,n,h,out_stride);CUDA(cudaGetLastError());
        }else if(mode==2){
            dense_gemm_tc<<<dim3((n+63)/64,(M+15)/16),128>>>(A,W,O,M,n,h);
        }else for(unsigned owner=0;owner<4;++owner){
            const unsigned r=owner*3;
            dense_gemm_tc<<<dim3((n+63)/64,1),128>>>(A+size_t(r)*h,W,O+size_t(r)*n,3,n,h);CUDA(cudaGetLastError());
        }
        CUDA(cudaGetLastError());
    }
};
static std::array<unsigned,2> top(const std::vector<uint16_t>& values,unsigned base,unsigned n){
    unsigned first=0,second=1;if(decode(values[base+second])>=decode(values[base+first]))std::swap(first,second);
    for(unsigned i=2;i<n;++i){const float x=decode(values[base+i]);if(x>=decode(values[base+first])){second=first;first=i;}else if(x>=decode(values[base+second]))second=i;}
    return {first,second}; // Highest output column wins BF16 ties.
}
static void check(Fixture& f,bool padded=false){
    std::array<std::vector<uint16_t>,MODES> v;
    for(unsigned mode=0;mode<MODES;++mode)v[mode]=f.out[mode]->read();
    for(unsigned mode=0;mode<MODES;++mode){
        const unsigned ld=f.ld(mode,padded);
        for(size_t i=0;i<v[mode].size();++i){
            const bool active=i/ld<M && i%ld<f.n;
            if(active)require(std::isfinite(decode(v[mode][i])),"unwritten/nonfinite output");
            else require(v[mode][i]==0xffff,"output stride/padding modified");
        }
        double maxdiff=0,diff2=0,norm2=0;size_t mismatches=0;
        for(unsigned r=0;r<M;++r)for(unsigned n=0;n<f.n;++n){
            const uint16_t base=v[0][size_t(r)*f.ld(0,padded)+n],other=v[mode][size_t(r)*ld+n];
            if(mode==1)require(base==other,"M3 batchm differs bitwise from scalar GEMVs");
            const double x=decode(base),d=std::abs(x-decode(other));
            require(d<=0.016+0.008*std::abs(x),"elementwise scalar difference exceeded bound");
            mismatches+=base!=other;maxdiff=std::max(maxdiff,d);diff2+=d*d;norm2+=x*x;
        }
        const double rel=std::sqrt(diff2/std::max(norm2,1e-30));require(rel<=0.003,"full relativeL2 exceeded .003");
        std::printf("{\"stage\":\"difference\",\"tiny\":%s,\"padded\":%s,\"mode\":%u,\"scalar_mismatches\":%zu,\"maxabs\":%.9g,\"relativeL2\":%.9g}\n",f.tiny?"true":"false",padded?"true":"false",mode,mismatches,maxdiff,rel);
    }
    double maxerr[MODES]={},err2[MODES]={},ref2=0;unsigned samples=0;
    for(unsigned r=0;r<M;++r){
        const auto expected=top(v[0],r*f.ld(0,padded),f.n);
        std::vector<unsigned> columns={0,1,f.n-1};
        for(unsigned mode=0;mode<MODES;++mode){
            const unsigned ld=f.ld(mode,padded);const auto t=top(v[mode],r*ld,f.n);
            columns.push_back(t[0]);columns.push_back(t[1]);
            std::printf("{\"stage\":\"top\",\"tiny\":%s,\"padded\":%s,\"row\":%u,\"mode\":%u,\"top\":%u,\"second\":%u,\"top_value\":%.9g,\"second_value\":%.9g,\"margin\":%.9g,\"top_fp64\":%.12g,\"second_fp64\":%.12g}\n",
                f.tiny?"true":"false",padded?"true":"false",r,mode,t[0],t[1],double(decode(v[mode][r*ld+t[0]])),double(decode(v[mode][r*ld+t[1]])),double(decode(v[mode][r*ld+t[0]])-decode(v[mode][r*ld+t[1]])),oracle(r,t[0],f.h,f.tiny),oracle(r,t[1],f.h,f.tiny));
            std::fflush(stdout);require(t[0]==expected[0],"top output column disagrees across implementations");
        }
        for(unsigned i=0;i<(f.tiny?f.n:128);++i)columns.push_back(f.tiny?i:mix(i*7919+r)%f.n);
        std::sort(columns.begin(),columns.end());columns.erase(std::unique(columns.begin(),columns.end()),columns.end());
        for(unsigned n:columns){
            const double ref=oracle(r,n,f.h,f.tiny);ref2+=ref*ref;++samples;
            for(unsigned mode=0;mode<MODES;++mode){
                const unsigned ld=f.ld(mode,padded);const double d=std::abs(decode(v[mode][r*ld+n])-ref);
                require(d<=0.008+0.004*std::abs(ref),"sampled FP64 error exceeded bound");
                if(f.tiny)require(d==0,"hand-computable tiny dot mismatch");
                maxerr[mode]=std::max(maxerr[mode],d);err2[mode]+=d*d;
            }
        }
    }
    for(unsigned mode=0;mode<MODES;++mode){
        const double e=std::sqrt(err2[mode]/std::max(ref2,1e-30));require(e<=0.004,"sampled FP64 relativeL2 exceeded .004");
        std::printf("{\"stage\":\"oracle\",\"mode\":%u,\"samples\":%u,\"maxabs\":%.9g,\"relativeL2\":%.9g}\n",mode,samples,maxerr[mode],e);
    }
    f.immutable();std::printf("{\"stage\":\"correctness\",\"tiny\":%s,\"padded\":%s,\"m3_scalar_bitwise\":true,\"pass\":true}\n",f.tiny?"true":"false",padded?"true":"false");std::fflush(stdout);
}
static bool timing(Fixture& f){
    for(unsigned repeat=0;repeat<20;++repeat)for(unsigned i=0;i<MODES;++i)f.launch((repeat+i)%MODES);
    CUDA(cudaDeviceSynchronize());cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
    std::vector<float> times[MODES];
    for(unsigned quartet=0;quartet<12;++quartet)for(unsigned i=0;i<MODES;++i){
        const unsigned mode=(quartet+i)%MODES;CUDA(cudaEventRecord(start));f.launch(mode);CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));
        float ms;CUDA(cudaEventElapsedTime(&ms,start,end));require(std::isfinite(ms)&&ms>0,"invalid timing");times[mode].push_back(ms);
    }
    CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));check(f);
    double median[MODES];for(unsigned mode=0;mode<MODES;++mode){
        auto sorted=times[mode];std::sort(sorted.begin(),sorted.end());median[mode]=(double(sorted[5])+sorted[6])/2;
        std::printf("{\"stage\":\"timing\",\"mode\":%u,\"warmup_each\":20,\"rotating_quartets\":12,\"median_ms\":%.9g,\"samples_ms\":[",mode,median[mode]);
        for(unsigned i=0;i<12;++i)std::printf("%s%.9g",i?",":"",times[mode][i]);std::printf("]}\n");
    }
    bool any=false;for(unsigned mode=1;mode<MODES;++mode){
        const double gain=median[0]/median[mode];any|=gain>=1.3;
        std::printf("{\"stage\":\"speed_gate\",\"mode\":%u,\"speedup\":%.9g,\"required\":1.3,\"pass\":%s}\n",mode,gain,gain>=1.3?"true":"false");
    }
    const double over_m3=std::min(median[1],median[3])/median[2];
    std::printf("{\"stage\":\"cohort_incremental_gate\",\"m12_vs_best_four_m3\":%.9g,\"required\":1.3,\"pass\":%s}\n",over_m3,over_m3>=1.3&&median[0]/median[2]>=1.3?"true":"false");
    return any;
}
static void run_checks(Fixture& f){
    f.immutable();
    for(bool padded:{false,true,false}){
        for(auto& o:f.out)o->poison();for(unsigned mode=0;mode<MODES;++mode)f.launch(mode,padded);
        CUDA(cudaDeviceSynchronize());check(f,padded);
    }
}
int main(int argc,char** argv){try{
    require(argc==2&&(std::string(argv[1])=="--run"||std::string(argv[1])=="--check"),"usage: bench --check|--run");
    cudaDeviceProp p;CUDA(cudaGetDeviceProperties(&p,0));require(p.major==12&&p.minor==1,"requires GB10");
    const size_t budget=size_t(V)*H*2+size_t(M)*H*2+MODES*size_t(M)*(V+16)*2+4+(MODES+3)*2*GUARD;
    require(budget<CAP,"preflight device budget");size_t free,total;CUDA(cudaMemGetInfo(&free,&total));require(free>=budget+RESERVE,"require allocation budget plus 4GiB free reserve");
    std::printf("{\"stage\":\"preflight\",\"M\":%u,\"N\":%u,\"K\":%u,\"budget_bytes\":%zu,\"free_bytes\":%zu,\"reserve_bytes\":%zu,\"cap_bytes\":%zu}\n",M,V,H,budget,free,RESERVE,CAP);
    for(unsigned mode=0;mode<MODES;++mode){cudaFuncAttributes a;
        if(mode==0)CUDA(cudaFuncGetAttributes(&a,dense_gemv_bf16));else if(mode==1)CUDA(cudaFuncGetAttributes(&a,dense_gemv_bf16_batchm));else CUDA(cudaFuncGetAttributes(&a,dense_gemm_tc));
        std::printf("{\"stage\":\"resources\",\"mode\":%u,\"regs\":%d,\"local_bytes\":%zu,\"shared_bytes\":%zu}\n",mode,a.numRegs,a.localSizeBytes,a.sharedSizeBytes);
    }
    {Fixture f(20,16,true);run_checks(f);}
    bool passed=true;{Fixture f(V,H,false);run_checks(f);if(std::string(argv[1])=="--run")passed=timing(f);}
    require(live==0,"allocation leak");const bool timed=std::string(argv[1])=="--run";
    std::printf("{\"stage\":\"complete\",\"peak_bytes\":%zu,\"correctness_pass\":true,\"timing_ran\":%s,\"any_mode_speed_pass\":%s}\n",peak,timed?"true":"false",timed?(passed?"true":"false"):"null");return passed?0:3;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 2;}}
