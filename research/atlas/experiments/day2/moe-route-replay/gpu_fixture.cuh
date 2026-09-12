// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "fixture.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <memory>
#define CK(call) do {auto status=(call);if(status!=cudaSuccess){std::fprintf(stderr,"%s:%d CUDA %s\n",__FILE__,__LINE__,cudaGetErrorString(status));std::exit(1);}}while(0)
static size_t live_bytes=0,peak_bytes=0;
template<class T> struct Buffer {
    T *raw=nullptr,*p=nullptr;size_t n,bytes;
    explicit Buffer(size_t n):n(n),bytes(n*sizeof(T)+512) {
        need(bytes<=8*GiB&&live_bytes<=8*GiB-bytes,"8GiB allocation cap");
        CK(cudaMalloc(&raw,bytes));p=reinterpret_cast<T*>(reinterpret_cast<char*>(raw)+256);
        CK(cudaMemset(raw,0xa5,bytes));live_bytes+=bytes;peak_bytes=std::max(peak_bytes,live_bytes);
    }
    ~Buffer(){cudaFree(raw);live_bytes-=bytes;}
    Buffer(const Buffer&)=delete;
    void put(const T* h){CK(cudaMemcpy(p,h,n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read()const{std::vector<T>x(n);CK(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
    void poison(cudaStream_t s){CK(cudaMemsetAsync(p,0xa5,n*sizeof(T),s));}
    void guards()const{
        std::array<unsigned char,256>a,b;CK(cudaMemcpy(a.data(),raw,256,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),p+n,256,cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<256;++i)need(a[i]==0xa5&&b[i]==0xa5,"redzone modified");
    }
};
__global__ void matrix_fill(unsigned char* w,size_t count,unsigned proj,unsigned n,unsigned k,bool scales) {
    unsigned div=scales?16:2;size_t stride=size_t(n)*k/div;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
        w[i]=weight_byte(proj,unsigned(i/stride),unsigned(i%n),unsigned((i%stride)/n),scales);
}
__global__ void matrix_shard(const unsigned char* full,unsigned char* shard,size_t count,unsigned proj,unsigned rank,bool scales) {
    unsigned div=scales?16:2,fn=proj==2?H:I,fk=proj==2?I:H;
    unsigned sn=proj==2?H:I/2,sk=proj==2?I/2:H;size_t ss=size_t(sn)*sk/div,fs=size_t(fn)*fk/div;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
        unsigned n=unsigned(i%sn)+(proj==2?0:rank*I/2),kp=unsigned((i%ss)/sn)+(proj==2?rank*I/(2*div):0);
        shard[i]=full[(i/ss)*fs+size_t(kp)*fn+n];
    }
}
__global__ void matrix_check(const unsigned char* data,const unsigned char* full,size_t count,unsigned proj,int rank,bool scales,unsigned* errors) {
    unsigned div=scales?16:2,fn=proj==2?H:I,fk=proj==2?I:H;
    unsigned sn=rank<0?fn:(proj==2?H:I/2),sk=rank<0?fk:(proj==2?I/2:H);size_t ss=size_t(sn)*sk/div,fs=size_t(fn)*fk/div;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
        unsigned n=unsigned(i%sn)+(rank<0||proj==2?0:unsigned(rank)*I/2);
        unsigned kp=unsigned((i%ss)/sn)+(rank<0||proj!=2?0:unsigned(rank)*I/(2*div));
        unsigned e=unsigned(i/ss);unsigned char v=weight_byte(proj,e,n,kp,scales);
        if(data[i]!=v||(rank>=0&&data[i]!=full[size_t(e)*fs+size_t(kp)*fn+n]))atomicOr(errors,1u);
    }
}
struct Matrix {
    unsigned proj,n,k;int rank;Buffer<unsigned char>w,s;
    Matrix(unsigned p,int r,const Matrix* full):proj(p),n(p==2?H:(r<0?I:I/2)),k(p==2?(r<0?I:I/2):H),rank(r),w(size_t(E)*n*k/2),s(size_t(E)*n*k/16) {
        if(r<0){matrix_fill<<<4096,256>>>(w.p,w.n,p,n,k,false);matrix_fill<<<4096,256>>>(s.p,s.n,p,n,k,true);}
        else{matrix_shard<<<4096,256>>>(full->w.p,w.p,w.n,p,unsigned(r),false);matrix_shard<<<4096,256>>>(full->s.p,s.p,s.n,p,unsigned(r),true);}
        CK(cudaGetLastError());
    }
    void check(const Matrix&full,Buffer<unsigned>&err){
        matrix_check<<<4096,256>>>(w.p,full.w.p,w.n,proj,rank,false,err.p);
        matrix_check<<<4096,256>>>(s.p,full.s.p,s.n,proj,rank,true,err.p);CK(cudaGetLastError());w.guards();s.guards();
    }
};
struct Tables {
    Buffer<unsigned long long>w{E},s{E};Buffer<float>s2{E};
    std::array<unsigned long long,E>expected_w{},expected_s{};std::array<float,E>expected_s2{};
    Tables(Matrix& m,int rank,bool tp){
        std::array<unsigned long long,E>a{},b{};std::array<float,E>c{};
        for(unsigned e=0;e<E;++e)if(tp||e/144==unsigned(rank)){
            a[e]=reinterpret_cast<unsigned long long>(m.w.p+size_t(e)*m.n*m.k/2);
            b[e]=reinterpret_cast<unsigned long long>(m.s.p+size_t(e)*m.n*m.k/16);c[e]=scale2_value(m.proj,e);
        }
        expected_w=a;expected_s=b;expected_s2=c;w.put(a.data());s.put(b.data());s2.put(c.data());
    }
    void guards(){w.guards();s.guards();s2.guards();
        need(w.read()==std::vector<unsigned long long>(expected_w.begin(),expected_w.end()),"packed pointer table immutable");
        need(s.read()==std::vector<unsigned long long>(expected_s.begin(),expected_s.end()),"scale pointer table immutable");
        need(s2.read()==std::vector<float>(expected_s2.begin(),expected_s2.end()),"scale2 table immutable");}
};
struct Weights {
    std::unique_ptr<Matrix>full[3],part[2][3];std::unique_ptr<Tables>table[2][2][3];
    Weights(){for(unsigned p=0;p<3;++p){full[p]=std::make_unique<Matrix>(p,-1,nullptr);
        for(int r=0;r<2;++r){part[r][p]=std::make_unique<Matrix>(p,r,full[p].get());table[0][r][p]=std::make_unique<Tables>(*full[p],r,false);table[1][r][p]=std::make_unique<Tables>(*part[r][p],r,true);}}
        CK(cudaDeviceSynchronize());check();}
    void check(){Buffer<unsigned>errors(1);CK(cudaMemset(errors.p,0,4));
        for(unsigned p=0;p<3;++p){full[p]->check(*full[p],errors);for(unsigned r=0;r<2;++r){part[r][p]->check(*full[p],errors);
            table[0][r][p]->guards();table[1][r][p]->guards();auto a=table[0][r][p]->s2.read(),b=table[1][r][p]->s2.read();
            for(unsigned e=0;e<E;++e)need(b[e]==scale2_value(p,e)&&(e/144!=r||a[e]==b[e]),"scale2 shard identity");}}
        CK(cudaDeviceSynchronize());need(errors.read()[0]==0,"all weight/scales exact bytes and immutable");std::puts("PASS all packed/scales shard bytes and scale2 identity");}
};
