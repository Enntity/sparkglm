// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "fixture.hpp"
#include "production/kernels/gb10/common/dense_gemm_bf16.cu"
#include "production/kernels/gb10/common/dense_gemv_bf16.cu"
#include "production/kernels/gb10/common/dense_gemv_bf16_batchm.cu"
#define CUDA(x) do {auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)
constexpr size_t GUARD=256,CAP=1400ull*1024*1024,RESERVE=4ull*1024*1024*1024;
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
    std::array<std::unique_ptr<Guarded>,3> out;
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
    void launch(unsigned mode,bool padded=false){
        require(mode<3 && (!padded || mode!=0),"launch mode");
        auto A=a.ptr<__nv_bfloat16>();auto W=w.ptr<__nv_bfloat16>();auto O=out[mode]->ptr<__nv_bfloat16>();
        const unsigned ld=padded?stride:n;
        if(mode==0)dense_gemm_bf16<<<dim3((n+15)/16,1),dim3(16,16)>>>(A,W,O,M,n,h);
        else if(mode==1)dense_gemv_bf16_batchm<<<(n+3)/4,256>>>(A,W,O,M,n,h,ld);
        else for(unsigned r=0;r<M;++r){dense_gemv_bf16<<<(n+3)/4,256>>>(A+size_t(r)*h,W,O+size_t(r)*ld,n,h);CUDA(cudaGetLastError());}
        CUDA(cudaGetLastError());
    }
};
static std::array<unsigned,2> top(const std::vector<uint16_t>& values,unsigned base,unsigned n){
    unsigned first=0,second=1;if(decode(values[base+second])>=decode(values[base+first]))std::swap(first,second);
    for(unsigned i=2;i<n;++i){const float x=decode(values[base+i]);if(x>=decode(values[base+first])){second=first;first=i;}else if(x>=decode(values[base+second]))second=i;}
    return {first,second}; // Highest token ID wins BF16 ties, matching argmax.
}
static void check(Fixture& f,bool padded=false){
    std::array<std::vector<uint16_t>,3> v;for(unsigned mode=0;mode<3;++mode)v[mode]=f.out[mode]->read();
    double maxdiff=0,diff2=0,norm2=0;size_t mismatches=0;
    for(unsigned mode=0;mode<3;++mode){
        const unsigned ld=padded&&mode?f.stride:f.n;
        for(size_t i=0;i<v[mode].size();++i){
            const bool active=i/ld<M && i%ld<f.n;
            if(active)require(std::isfinite(decode(v[mode][i])),"unwritten/nonfinite output");
            else require(v[mode][i]==0xffff,"output stride/padding modified");
        }
    }
    for(unsigned r=0;r<M;++r)for(unsigned n=0;n<f.n;++n){
        const size_t base=size_t(r)*f.n+n, other=size_t(r)*(padded?f.stride:f.n)+n;
        require(v[1][other]==v[2][other],"batchm differs bitwise from three scalar GEMVs");
        double x=decode(v[0][base]),y=decode(v[1][other]),d=std::abs(x-y);
        require(d<=0.016+0.008*std::abs(x),"GEMM/batchm elementwise difference exceeded bound");
        mismatches+=v[0][base]!=v[1][other];maxdiff=std::max(maxdiff,d);diff2+=d*d;norm2+=x*x;
    }
    const double rel=std::sqrt(diff2/std::max(norm2,1e-30));require(rel<=0.003,"GEMM/batchm relativeL2 exceeded .003");
    double maxerr[3]={},err2[3]={},ref2=0;unsigned samples=0;
    for(unsigned r=0;r<M;++r){
        const auto expected=top(v[0],r*f.n,f.n);
        for(unsigned mode=0;mode<3;++mode){
            const unsigned ld=padded&&mode?f.stride:f.n;auto t=top(v[mode],r*ld,f.n);
            require(t[0]==expected[0],"top token disagrees across implementations");
            std::printf("{\"stage\":\"top\",\"tiny\":%s,\"padded\":%s,\"row\":%u,\"mode\":%u,\"top\":%u,\"second\":%u,\"margin\":%.9g,\"top_fp64\":%.12g,\"second_fp64\":%.12g}\n",
                f.tiny?"true":"false",padded?"true":"false",r,mode,t[0],t[1],double(decode(v[mode][r*ld+t[0]])-decode(v[mode][r*ld+t[1]])),oracle(r,t[0],f.h,f.tiny),oracle(r,t[1],f.h,f.tiny));
        }
        std::vector<unsigned> columns={0,1,f.n-1,expected[0],expected[1]};
        for(unsigned i=0;i<(f.tiny?f.n:128);++i)columns.push_back(f.tiny?i:mix(i*7919+r)%f.n);
        for(unsigned n:columns){
            const double ref=oracle(r,n,f.h,f.tiny);ref2+=ref*ref;++samples;
            for(unsigned mode=0;mode<3;++mode){
                const unsigned ld=padded&&mode?f.stride:f.n;const double d=std::abs(decode(v[mode][r*ld+n])-ref);
                require(d<=0.008+0.004*std::abs(ref),"sampled FP64 error exceeded bound");
                if(f.tiny)require(d==0,"hand-computable tiny dot mismatch");
                maxerr[mode]=std::max(maxerr[mode],d);err2[mode]+=d*d;
            }
        }
    }
    for(unsigned mode=0;mode<3;++mode){const double e=std::sqrt(err2[mode]/std::max(ref2,1e-30));require(e<=0.004,"sampled FP64 relativeL2 exceeded .004");
        std::printf("{\"stage\":\"oracle\",\"mode\":%u,\"samples\":%u,\"maxabs\":%.9g,\"relativeL2\":%.9g}\n",mode,samples,maxerr[mode],e);}
    f.immutable();std::printf("{\"stage\":\"correctness\",\"tiny\":%s,\"padded\":%s,\"gemm_batchm_mismatches\":%zu,\"maxabs\":%.9g,\"relativeL2\":%.9g,\"batchm_threegemv_bitwise\":true,\"pass\":true}\n",f.tiny?"true":"false",padded?"true":"false",mismatches,maxdiff,rel);std::fflush(stdout);
}
static bool timing(Fixture& f){
    for(unsigned repeat=0;repeat<20;++repeat)for(unsigned i=0;i<3;++i)f.launch((repeat+i)%3);
    CUDA(cudaDeviceSynchronize());cudaEvent_t start,end;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&end));
    std::vector<float> times[3];
    for(unsigned pair=0;pair<9;++pair)for(unsigned i=0;i<3;++i){
        const unsigned mode=(pair+i)%3;CUDA(cudaEventRecord(start));f.launch(mode);CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));
        float ms;CUDA(cudaEventElapsedTime(&ms,start,end));require(std::isfinite(ms)&&ms>0,"invalid timing");times[mode].push_back(ms);
    }
    CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(end));check(f);
    double median[3];for(unsigned mode=0;mode<3;++mode){auto sorted=times[mode];std::sort(sorted.begin(),sorted.end());median[mode]=sorted[4];
        std::printf("{\"stage\":\"timing\",\"mode\":%u,\"warmup_each\":20,\"rotating_triplets\":9,\"median_ms\":%.9g,\"samples_ms\":[",mode,median[mode]);
        for(unsigned i=0;i<9;++i)std::printf("%s%.9g",i?",":"",times[mode][i]);std::printf("]}\n");}
    const double speedup=median[0]/median[1];std::printf("{\"stage\":\"speed_gate\",\"speedup\":%.9g,\"required\":1.3,\"pass\":%s}\n",speedup,speedup>=1.3?"true":"false");return speedup>=1.3;
}
int main(int argc,char** argv){try{
    require(argc==2&&(std::string(argv[1])=="--run"||std::string(argv[1])=="--check"),"usage: bench --check|--run");
    cudaDeviceProp p;CUDA(cudaGetDeviceProperties(&p,0));require(p.major==12&&p.minor==1,"requires GB10");
    const size_t budget=size_t(V)*H*2+size_t(M)*H*2+3*size_t(M)*(V+16)*2+4+6*2*GUARD;
    require(budget<CAP,"preflight device budget");size_t free,total;CUDA(cudaMemGetInfo(&free,&total));require(free>=budget+RESERVE,"require allocation budget plus 4GiB free reserve");
    std::printf("{\"stage\":\"preflight\",\"budget_bytes\":%zu,\"free_bytes\":%zu,\"reserve_bytes\":%zu,\"cap_bytes\":%zu}\n",budget,free,RESERVE,CAP);
    for(auto mode:{0,1,2}){cudaFuncAttributes a;if(mode==0)CUDA(cudaFuncGetAttributes(&a,dense_gemm_bf16));else if(mode==1)CUDA(cudaFuncGetAttributes(&a,dense_gemv_bf16_batchm));else CUDA(cudaFuncGetAttributes(&a,dense_gemv_bf16));
        std::printf("{\"stage\":\"resources\",\"mode\":%d,\"regs\":%d,\"local_bytes\":%zu,\"shared_bytes\":%zu}\n",mode,a.numRegs,a.localSizeBytes,a.sharedSizeBytes);}
    {Fixture f(20,16,true);for(auto& o:f.out)o->poison();for(unsigned mode=0;mode<3;++mode)f.launch(mode);CUDA(cudaDeviceSynchronize());check(f);
        for(unsigned mode=1;mode<3;++mode){f.out[mode]->poison();f.launch(mode,true);}CUDA(cudaDeviceSynchronize());check(f,true);}
    bool passed=true;{Fixture f(V,H,false);for(auto& o:f.out)o->poison();for(unsigned mode=0;mode<3;++mode)f.launch(mode);CUDA(cudaDeviceSynchronize());check(f);
        for(unsigned mode=1;mode<3;++mode){f.out[mode]->poison();f.launch(mode,true);}CUDA(cudaDeviceSynchronize());check(f,true);
        // Restore packed output layout before timing; padding remains poison.
        for(auto& o:f.out)o->poison();for(unsigned mode=0;mode<3;++mode)f.launch(mode);CUDA(cudaDeviceSynchronize());check(f);
        if(std::string(argv[1])=="--run")passed=timing(f);}
    require(live==0,"allocation leak");std::printf("{\"stage\":\"complete\",\"peak_bytes\":%zu,\"pass\":%s}\n",peak,passed?"true":"false");return passed?0:3;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 2;}}
