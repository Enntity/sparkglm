// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
constexpr unsigned H=4096,I=2048,E=288,M=3,TOP=8,R=M*TOP;
constexpr size_t GiB=size_t(1)<<30;
inline void need(bool ok,const char* why) {
    if(!ok){std::fprintf(stderr,"FAIL %s\n",why);std::exit(2);}
}
HD inline unsigned hash32(unsigned x) {
    x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);
}
// Coordinate-derived canonical row-major values; storage is T [K/2,N].
HD inline unsigned char weight_byte(unsigned proj,unsigned e,unsigned n,unsigned kp,bool scale) {
    unsigned x=hash32(137u+proj*7919u+e*4099u+n*6197u+kp*104729u);
    return scale?static_cast<unsigned char>(0x20u+x%16u):static_cast<unsigned char>(x);
}
HD inline float scale2_value(unsigned proj,unsigned e) {return .125f+float((proj*7+e)%8)*.015625f;}
inline float f32(uint32_t bits){float f;std::memcpy(&f,&bits,4);return f;}
inline uint32_t bits32(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
inline float bf(uint16_t b){return f32(uint32_t(b)<<16);}
inline uint16_t round_bf(float f){uint32_t b=bits32(f);return uint16_t((b+0x7fffu+((b>>16)&1u))>>16);}
inline double e4(unsigned b) {
    unsigned e=(b>>3)&15,m=b&7;
    return (b&128?-1.0:1.0)*std::ldexp(e?1.0+m/8.0:m/8.0,e?int(e)-7:-6);
}
inline double e2(unsigned b){constexpr double t[8]={0,.5,1,1.5,2,3,4,6};return(b&8?-1:1)*t[b&7];}
inline double w_value(unsigned proj,unsigned e,unsigned n,unsigned k) {
    auto p=weight_byte(proj,e,n,k/2,false);
    return e2((p>>((k&1)*4))&15)*e4(weight_byte(proj,e,n,k/16,true))*scale2_value(proj,e);
}
struct Routes {
    std::array<unsigned,R> ids{};
    std::array<float,R> weights{};
    std::array<uint16_t,M*H> input{};
    Routes(bool skew,unsigned owner) {
        for(unsigned t=0;t<M;++t) {
            float sum=0;
            for(unsigned j=0;j<TOP;++j) {
                unsigned half=skew?(j==7?1:0):(j&1);
                if(skew&&(owner&1))half^=1;
                // Owner-local hot groups recur across rows; balanced groups vary per token.
                unsigned local=skew?(owner*13+j*11)%144:(owner*29+t*19+(j/2)*31)%144;
                ids[t*TOP+j]=half*144+local;weights[t*TOP+j]=float(j+1);sum+=float(j+1);
            }
            for(unsigned j=0;j<TOP;++j)weights[t*TOP+j]/=sum;
            for(unsigned k=0;k<H;++k)input[t*H+k]=round_bf(float(int(hash32(owner*9001+t*331+k)%1025)-512)/512);
        }
    }
};
inline size_t shard_index(unsigned proj,unsigned rank,unsigned n,unsigned kp,bool scale) {
    unsigned fn=proj==2?H:I, fk=proj==2?I:H;
    unsigned sn=proj==2?H:I/2,sk=proj==2?I/2:H,div=scale?16:2;
    need(n<sn&&kp<sk/div,"shard coordinate bounds");
    unsigned gn=n+(proj==2?0:rank*I/2),gkp=kp+(proj==2?rank*I/(2*div):0);
    need(gn<fn&&gkp<fk/div,"full coordinate bounds");return size_t(gkp)*fn+gn;
}
inline double median(std::vector<double> a){need(!a.empty(),"nonempty timings");std::sort(a.begin(),a.end());return a[a.size()/2];}
inline void host_tests() {
    for(bool skew:{false,true})for(unsigned owner=0;owner<4;++owner) {
        Routes r(skew,owner);
        for(unsigned t=0;t<M;++t) {
            std::array<bool,E> seen{};unsigned half=0;double sum=0;
            for(unsigned j=0;j<TOP;++j){unsigned e=r.ids[t*TOP+j];need(e<E&&!seen[e],"unique top8");seen[e]=true;half+=e<144;sum+=r.weights[t*TOP+j];}
            need(half==(skew?((owner&1)?1u:7u):4u),"declared ownership balance");need(std::abs(sum-1)<1e-6,"normalized routes");
        }
    }
    // Independently enumerate smaller row/column splits, including both shard boundaries.
    for(unsigned proj=0;proj<3;++proj)for(unsigned rank=0;rank<2;++rank)for(bool s:{false,true}) {
        unsigned n=proj==2?H:I/2,k=(proj==2?I/2:H)/(s?16:2),fn=proj==2?H:I;
        for(unsigned nn:{0u,n-1})for(unsigned kk:{0u,k-1}) {
            size_t i=shard_index(proj,rank,nn,kk,s);
            need(i%fn==nn+(proj==2?0:rank*I/2),"N shard identity");
            need(i/fn==kk+(proj==2?rank*k:0),"K shard identity");
        }
    }
    need(e4(0x38)==1&&e4(0x08)==std::ldexp(1.,-6)&&e2(7)==6&&e2(15)==-6,"quant decode known values");
    need(round_bf(1.00390625f)==0x3f80&&round_bf(1.01171875f)==0x3f82,"BF16 ties-to-even");
    need(2*(size_t(E)*3*I*H*9/16)==7776*(size_t(1)<<20),"weight payload budget");
    std::puts("PASS host: 8 routing fixtures, shard boundaries, exact decode/BF16 ties, memory formula");
}
