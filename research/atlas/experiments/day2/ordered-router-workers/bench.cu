// SPDX-License-Identifier: AGPL-3.0-only
// Private standalone harness; worker header selected at compile time, unchanged.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>
#include "production/dense_gemm_bf16.cu"
#undef TILE_M
#undef TILE_N
#undef TILE_K
#ifndef CANDIDATE_HEADER
#error CANDIDATE_HEADER must select a separately reviewed unchanged worker header
#endif
#include CANDIDATE_HEADER
#ifndef CANDIDATE_KERNEL
#error CANDIDATE_KERNEL must name the reviewed single kernel for attribute reporting
#endif
constexpr unsigned M=3,N=288,K=4096,GUARD=256;
constexpr size_t CAP=32ull<<20,RESERVE=4ull<<30;
size_t live=0,peak=0;
void need(bool ok,const char*why){if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(2);}}
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"FAIL CUDA line=%u %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
float fp32(uint16_t b){uint32_t v=uint32_t(b)<<16;float f;std::memcpy(&f,&v,4);return f;}
uint16_t bf16(float f){uint32_t b;std::memcpy(&b,&f,4);b+=0x7fffu+((b>>16)&1u);return uint16_t(b>>16);}
struct Buffer{
    unsigned char*raw=nullptr;__nv_bfloat16*p=nullptr;size_t count,bytes;
    explicit Buffer(size_t n):count(n),bytes(n*2+2*GUARD){
        need(live+bytes<=CAP,"32MiB explicit device cap");CK(cudaMalloc(&raw,bytes));p=reinterpret_cast<__nv_bfloat16*>(raw+GUARD);
        CK(cudaMemset(raw,0xA5,bytes));live+=bytes;peak=std::max(peak,live);
    }
    ~Buffer(){cudaFree(raw);live-=bytes;}
    void put(const std::vector<uint16_t>&v){need(v.size()==count,"input size");CK(cudaMemcpy(p,v.data(),count*2,cudaMemcpyHostToDevice));}
    std::vector<uint16_t>get(){std::vector<uint16_t>v(count);CK(cudaMemcpy(v.data(),p,count*2,cudaMemcpyDeviceToHost));return v;}
    void poison(uint16_t bits){put(std::vector<uint16_t>(count,bits));}
    void guards(){std::array<unsigned char,GUARD>a,b;CK(cudaMemcpy(a.data(),raw,GUARD,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),raw+GUARD+count*2,GUARD,cudaMemcpyDeviceToHost));for(unsigned i=0;i<GUARD;++i)need(a[i]==0xA5&&b[i]==0xA5,"256-byte redzone");}
};
struct Clock{
    cudaEvent_t a,b;Clock(){CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));}~Clock(){cudaEventDestroy(a);cudaEventDestroy(b);}
    template<class F>double run(cudaStream_t s,F f){CK(cudaEventRecord(a,s));f();CK(cudaEventRecord(b,s));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));need(std::isfinite(ms)&&ms>0,"finite positive timing");return ms;}
};
void reference(Buffer&a,Buffer&b,Buffer&c,cudaStream_t s){dense_gemm_bf16<<<dim3(18,1),dim3(16,16),0,s>>>(a.p,b.p,c.p,M,N,K);CK(cudaGetLastError());}
void candidate(Buffer&a,Buffer&b,Buffer&c,cudaStream_t s){CK(launch_ordered_bf16_3x288(a.p,b.p,c.p,s));}
struct Fixture{
    const char*name;std::vector<uint16_t>a=std::vector<uint16_t>(M*K),b=std::vector<uint16_t>(N*K);
    explicit Fixture(unsigned id){
        const char*names[]={"signed_zero","one_hot","constant","alternating","small_integer","general_scaled","order_adversary","bf16_ties","tiny_subnormal","all_zero"};name=names[id];
        uint32_t seed=0x51912u;auto rng=[&]{seed=1664525u*seed+1013904223u;return seed;};
        for(unsigned m=0;m<M;++m)for(unsigned k=0;k<K;++k){float v=0;
            if(id==0){a[m*K+k]=((m+k)&1)?0x8000:0;continue;}
            if(id==1)v=k==m?1:0;
            if(id==2)v=(m+1)*.5f;
            if(id==3)v=(k&1)?-1:1;
            if(id==4)v=int((k+m)%5)-2;
            if(id==5)v=std::ldexp(float(int(rng()%65)-32),-int(3+rng()%13));
            if(id==6)v=k<1024?4:k<3072?1.f/64:4;
            if(id==7)v=k<3?1:0;
            if(id==8){a[m*K+k]=uint16_t(((m+k)%2?0x8000:0)| (1+(k%127)));continue;}
            a[m*K+k]=bf16(v);
        }
        for(unsigned n=0;n<N;++n)for(unsigned k=0;k<K;++k){float v=0;
            if(id==0){b[n*K+k]=((n+k)&1)?0x8000:0;continue;}
            if(id==1)v=float(int((n+k)%7)-3);
            if(id==2)v=(int(n%5)-2)*.5f;
            if(id==3)v=1;
            if(id==4)v=int((k+n)%7)-3;
            if(id==5)v=std::ldexp(float(int(rng()%65)-32),-int(3+rng()%13));
            if(id==6)v=(k<1024?4:k<3072?1.f/64:-4)*(n%2?-1:1);
            if(id==7)v=k==0?1:k==1?1.f/256:k==2?std::ldexp(float(int(n%3)-1),-20):0;
            if(id==8)v=(n%2?-1.f:1.f);
            b[n*K+k]=bf16(v);
        }
    }
};
std::array<unsigned,8>top(const std::vector<uint16_t>&v,unsigned m){std::array<unsigned,N>ids;std::iota(ids.begin(),ids.end(),0);std::sort(ids.begin(),ids.end(),[&](unsigned a,unsigned b){float x=fp32(v[m*N+a]),y=fp32(v[m*N+b]);return x>y||(x==y&&a<b);});std::array<unsigned,8>r;std::copy_n(ids.begin(),8,r.begin());return r;}
std::vector<uint16_t>oracle(const Fixture&f){
    std::vector<uint16_t>out(M*N);const double u=std::ldexp(1.,-24),gamma=(2*K*u)/(1-2*K*u);
    for(unsigned m=0;m<M;++m)for(unsigned n=0;n<N;++n){float acc=0;double exact=0,sumabs=0;
        for(unsigned k=0;k<K;++k){float x=fp32(f.a[m*K+k]),y=fp32(f.b[n*K+k]);
            volatile float product=x*y;volatile float next=acc+product;acc=next;
            const double term=double(x)*double(y);exact+=term;sumabs+=std::abs(term);
        }
        out[m*N+n]=bf16(acc);double got=fp32(out[m*N+n]);
        double ulp=std::max(std::ldexp(1.,-133),got==0?0.:std::ldexp(1.,std::ilogb(std::abs(got))-7));
        double bound=gamma*sumabs+ulp+2*K*std::ldexp(1.,-149);
        need(std::isfinite(got)&&std::abs(got-exact)<=bound,"independent FP64 gamma bound");
        if(std::strcmp(f.name,"signed_zero")==0||std::strcmp(f.name,"alternating")==0||std::strcmp(f.name,"all_zero")==0||std::strcmp(f.name,"order_adversary")==0)need(out[m*N+n]==0,"hand zero oracle");
        if(std::strcmp(f.name,"constant")==0||std::strcmp(f.name,"small_integer")==0)need(got==exact,"hand exact integer/power-of-two dot");
        if(std::strcmp(f.name,"bf16_ties")==0)need(out[m*N+n]==bf16(n%3==2?1.0078125f:1.f),"hand BF16 tie oracle");
        if(std::strcmp(f.name,"one_hot")==0)need(out[m*N+n]==f.b[n*K+m],"hand one-hot oracle");
    }
    return out;
}
void compare(const char*label,const Fixture&f,const std::vector<uint16_t>&expected,const std::vector<uint16_t>&actual){
    for(unsigned i=0;i<M*N;++i)if(!std::isfinite(fp32(actual[i]))||expected[i]!=actual[i]){
        std::fprintf(stderr,"FAIL fixture=%s path=%s row=%u col=%u expected=%04x actual=%04x\n",f.name,label,i/N,i%N,expected[i],actual[i]);std::exit(2);}
}
void verify(Fixture&f,Buffer&a,Buffer&b,Buffer&ref,Buffer&out,cudaStream_t s){
    a.put(f.a);b.put(f.b);auto expected=oracle(f);
    for(auto poison:{uint16_t(0x7fc1),uint16_t(0xffc3)}){
        ref.poison(poison);out.poison(poison);reference(a,b,ref,s);candidate(a,b,out,s);CK(cudaStreamSynchronize(s));
        auto r=ref.get(),c=out.get();compare("baseline_cpu",f,expected,r);compare("candidate_baseline",f,r,c);
        for(unsigned m=0;m<M;++m)need(top(r,m)==top(c,m)&&top(expected,m)==top(c,m),"all3 rows deterministic top8");
        need(a.get()==f.a&&b.get()==f.b,"inputs unchanged");a.guards();b.guards();ref.guards();out.guards();
    }
    std::printf("PASS fixture=%s outputs=864 poisons=2 ordered_cpu=864 fp64=864 top8_rows=3\n",f.name);
}
void attributes(){cudaFuncAttributes a,b;CK(cudaFuncGetAttributes(&a,dense_gemm_bf16));CK(cudaFuncGetAttributes(&b,CANDIDATE_KERNEL));
    need(b.maxThreadsPerBlock>=1&&b.sharedSizeBytes<=24*1024,"candidate static shared limit");
    std::printf("ATTR baseline_regs=%d baseline_static_shared=%zu candidate_regs=%d candidate_static_shared=%zu candidate_local_bytes=%zu\n",a.numRegs,a.sharedSizeBytes,b.numRegs,b.sharedSizeBytes,b.localSizeBytes);
}
int main(){
    need(std::fesetround(FE_TONEAREST)==0,"CPU round-to-nearest");volatile float tiny=std::numeric_limits<float>::denorm_min();volatile float one=1;volatile float under=tiny*one;need(under==tiny&&under>0,"CPU gradual underflow");
    size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=CAP+RESERVE,"32MiB cap plus4GiB initial free reserve");attributes();
    Buffer a(M*K),b(N*K),ref(M*N),out(M*N);CK(cudaMemGetInfo(&free,&total));need(free>=RESERVE,"4GiB actual free reserve");
    cudaStream_t s;CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    const auto original_a=a.get(),original_b=b.get();
    for(unsigned mask=1;mask<8;++mask){out.poison(0x7fc1);CK(cudaGetLastError());
        need(launch_ordered_bf16_3x288(mask&1?nullptr:a.p,mask&2?nullptr:b.p,mask&4?nullptr:out.p,s)==cudaErrorInvalidValue,"null host rejection");
        CK(cudaStreamSynchronize(s));need(out.get()==std::vector<uint16_t>(M*N,0x7fc1),"null rejection unchanged output");need(a.get()==original_a&&b.get()==original_b,"null rejection inputs unchanged");a.guards();b.guards();out.guards();}
    for(unsigned i=0;i<10;++i){Fixture f(i);verify(f,a,b,ref,out,s);}
    Fixture f(5);a.put(f.a);b.put(f.b);for(unsigned i=0;i<20;++i){reference(a,b,ref,s);candidate(a,b,out,s);}CK(cudaStreamSynchronize(s));
    Clock clock;std::vector<double>r,c;
    for(unsigned i=0;i<15;++i){double rm,cm;if(i%2==0){rm=clock.run(s,[&]{reference(a,b,ref,s);});cm=clock.run(s,[&]{candidate(a,b,out,s);});}else{cm=clock.run(s,[&]{candidate(a,b,out,s);});rm=clock.run(s,[&]{reference(a,b,ref,s);});}r.push_back(rm);c.push_back(cm);std::printf("PAIR pair=%u reference_ms=%.9f candidate_ms=%.9f\n",i,rm,cm);}
    need(a.get()==f.a&&b.get()==f.b,"timed inputs unchanged");a.guards();b.guards();ref.guards();out.guards();compare("timed",f,ref.get(),out.get());
    std::sort(r.begin(),r.end());std::sort(c.begin(),c.end());double gain=r[7]/c[7];
    std::printf("RESULT reference_ms=%.9f candidate_ms=%.9f speedup=%.9f threshold=2.0 peak_bytes=%zu null_cases=7 fixtures=10\n",r[7],c[7],gain,peak);
    CK(cudaStreamDestroy(s));return gain>=2?0:3;
}
