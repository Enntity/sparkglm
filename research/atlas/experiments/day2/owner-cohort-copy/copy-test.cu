// SPDX-License-Identifier: AGPL-3.0-only
// Independent contract test. Compile only with a parent-frozen candidate.
#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
extern "C" int copy_four_segments(void*,void*,void*,void*,void*,size_t,int,void*);
constexpr size_t MiB=1024*1024, CAP=256*MiB, RESERVE=size_t(4)*1024*MiB, GUARD=256;
static size_t live=0,peak=0;
static void need(bool b,const char* msg){if(!b){std::fprintf(stderr,"FAIL %s\n",msg);std::exit(2);}}
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);}}while(0)
struct Call{std::array<void*,5> p;size_t n;int scatter;std::string name;};
static int invoke(const Call& c,cudaStream_t s){return copy_four_segments(c.p[0],c.p[1],c.p[2],c.p[3],c.p[4],c.n,c.scatter,reinterpret_cast<void*>(s));}
static std::vector<Call> invalid_calls(std::array<void*,5> p){
    Call base{p,96,0,""};std::vector<Call> out;
    auto add=[&](const char* name,auto change){Call c=base;c.name=name;change(c);out.push_back(c);};
    add("zero",[](Call&c){c.n=0;});add("size17",[](Call&c){c.n=17;});
    add("oversize",[](Call&c){c.n=16*MiB+16;});add("size_overflow",[](Call&c){c.n=SIZE_MAX&~size_t(15);});
    add("scatter2",[](Call&c){c.scatter=2;});add("scatter_negative",[](Call&c){c.scatter=-1;});
    for(unsigned i=0;i<5;++i){
        add("null",[&](Call&c){c.p[i]=nullptr;});
        add("pointer_misaligned",[&](Call&c){c.p[i]=static_cast<unsigned char*>(c.p[i])+1;});
        add("range_overflow",[&](Call&c){c.p[i]=reinterpret_cast<void*>(UINTPTR_MAX-15);});
    }
    for(unsigned i=0;i<5;++i)for(unsigned j=i+1;j<5;++j){
        add("equal_ranges",[&](Call&c){c.p[j]=c.p[i];});
        add("overlap16",[&](Call&c){c.p[j]=static_cast<unsigned char*>(c.p[i])+16;});
    }
    // Segment overlaps only the packed range's LAST quarter: packed extent must be4*n.
    add("packed_last_quarter",[](Call&c){c.p[0]=static_cast<unsigned char*>(c.p[4])+3*c.n;});
    return out;
}
static void host_invalid(){
    alignas(16) std::array<unsigned char,2048> data,copy;data.fill(0x6d);copy=data;
    std::array<void*,5> p={data.data(),data.data()+256,data.data()+512,data.data()+768,data.data()+1024};
    auto cases=invalid_calls(p);
    for(auto c:cases)for(int direction:{0,1}){
        if(c.scatter==0)c.scatter=direction;
        need(invoke(c,nullptr)==int(cudaErrorInvalidValue),c.name.c_str());
        need(data==copy,"invalid host arguments changed backing bytes");
    }
    std::printf("PASS HOST_INVALID cases=%zu directions=2 backing_unchanged=true\n",cases.size());
}
struct Block{
    unsigned char *raw=nullptr,*p=nullptr;size_t n,total;
    explicit Block(size_t n):n(n),total(n+2*GUARD){
        need(total<=CAP&&live<=CAP-total,"256MiB allocation cap");
        CK(cudaMalloc(&raw,total));p=raw+GUARD;live+=total;peak=std::max(peak,live);
        CK(cudaMemset(raw,0xa5,total));
    }
    ~Block(){cudaFree(raw);live-=total;}
    template<class F>void fill(F f){
        std::array<unsigned char,65536> buf;
        for(size_t start=0;start<n;start+=buf.size()){
            size_t count=std::min(buf.size(),n-start);for(size_t i=0;i<count;++i)buf[i]=f(start+i);
            CK(cudaMemcpy(p+start,buf.data(),count,cudaMemcpyHostToDevice));
        }
    }
    template<class F>void check(F f,const char* label){
        std::array<unsigned char,65536> buf;
        for(size_t start=0;start<n;start+=buf.size()){
            size_t count=std::min(buf.size(),n-start);CK(cudaMemcpy(buf.data(),p+start,count,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<count;++i)if(buf[i]!=f(start+i)){
                std::fprintf(stderr,"FAIL BYTE %s offset=%zu got=%u expected=%u\n",label,start+i,unsigned(buf[i]),unsigned(f(start+i)));std::exit(2);}
        }
        std::array<unsigned char,GUARD> a,b;CK(cudaMemcpy(a.data(),raw,GUARD,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(),p+n,GUARD,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<GUARD;++i)need(a[i]==0xa5&&b[i]==0xa5,"redzone changed");
    }
};
static unsigned char pattern(unsigned segment,size_t offset,unsigned seed){
    uint64_t v=uint64_t(offset)*0x9e3779b97f4a7c15ULL+uint64_t(segment+1)*0xbf58476d1ce4e5b9ULL+seed;
    v^=v>>31;v*=0x94d049bb133111ebULL;return static_cast<unsigned char>((v^(v>>33))>>19);
}
static void valid(size_t n,bool adjacent,int scatter,cudaStream_t stream){
    std::vector<std::unique_ptr<Block>> segments;
    for(unsigned i=0;i<(adjacent?1u:4u);++i)segments.emplace_back(std::make_unique<Block>(adjacent?4*n:n));
    Block packed(4*n);std::array<void*,5> p;
    for(unsigned i=0;i<4;++i)p[i]=segments[adjacent?0:i]->p+(adjacent?i*n:0);p[4]=packed.p;
    auto source=[&](size_t i){return pattern(unsigned(i/n),i%n,scatter?71:29);};
    for(unsigned i=0;i<segments.size();++i)segments[i]->fill([&](size_t j){return scatter?0x37:source(adjacent?j:i*n+j);});
    packed.fill([&](size_t j){return scatter?source(j):0xc9;});
    size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=RESERVE,"4GiB free reserve after allocation");
    Call c{p,n,scatter,"valid"};need(invoke(c,stream)==int(cudaSuccess),"valid copy rejected");CK(cudaStreamSynchronize(stream));
    for(unsigned i=0;i<segments.size();++i)segments[i]->check([&](size_t j){return source(adjacent?j:i*n+j);},"segments");
    packed.check(source,"packed");
    std::printf("PASS COPY segment_bytes=%zu adjacent=%d scatter=%d default_stream=%d bytes=%zu\n",n,adjacent,scatter,stream==nullptr,8*n);
}
static void gpu_invalid(cudaStream_t stream){
    std::vector<std::unique_ptr<Block>> b;std::array<void*,5> p;
    for(unsigned i=0;i<5;++i){b.emplace_back(std::make_unique<Block>(i==4?384:96));p[i]=b.back()->p;b.back()->fill([&](size_t j){return pattern(i,j,83);});}
    auto cases=invalid_calls(p);
    for(auto c:cases)for(int direction:{0,1}){
        if(c.scatter==0)c.scatter=direction;
        need(invoke(c,stream)==int(cudaErrorInvalidValue),c.name.c_str());CK(cudaStreamSynchronize(stream));
        for(unsigned i=0;i<5;++i)b[i]->check([&](size_t j){return pattern(i,j,83);},"invalid unchanged");
    }
    std::printf("PASS GPU_INVALID cases=%zu directions=2 backing_unchanged=true\n",cases.size());
}
int main(int argc,char**argv){
    need(argc==2,"use --host-invalid or --gpu");
    if(std::strcmp(argv[1],"--host-invalid")==0){host_invalid();return 0;}
    need(std::strcmp(argv[1],"--gpu")==0,"unknown mode");host_invalid();
    size_t free,total;CK(cudaMemGetInfo(&free,&total));need(free>=CAP+RESERVE,"256MiB cap plus4GiB initial reserve");
    cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));gpu_invalid(stream);
    for(size_t n:{size_t(16),size_t(96),size_t(24576),size_t(16)*MiB})for(bool adjacent:{true,false})for(int scatter:{0,1}){
        valid(n,adjacent,scatter,stream);
        if(n<=96)valid(n,adjacent,scatter,nullptr);
    }
    CK(cudaStreamDestroy(stream));need(live==0,"all test allocations released");
    std::printf("PASS COMPLETE peak_device_bytes=%zu cap_bytes=%zu max_vectors=%zu requires_gridstride=true\n",peak,CAP,size_t(4)*16*MiB/16);
}
