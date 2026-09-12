// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#ifdef __CUDACC__
#define BOTH __host__ __device__
#else
#define BOTH
#endif
constexpr unsigned M=3, H=4096, V=154856;
BOTH inline uint32_t mix(uint32_t x) {
    x ^= x>>16; x*=0x7feb352du; x ^= x>>15; x*=0x846ca68bu; return x^(x>>16);
}
// Integer construction fixes the ORIGINAL BF16 payload, independent of GPU
// rounding or host FP operations. Three nonconstant activation rows; weights
// have independent signs/mantissas and six exponents (roughly .001.. .06).
BOTH inline uint16_t a_bits(unsigned row,unsigned k,bool tiny) {
    if(tiny) return k==0 ? uint16_t(row==0?0x3f80:row==1?0x4000:0x4040)
                       : k==1?uint16_t(0xbf00):uint16_t(0);
    const uint32_t x=mix(0x582d5ab1u+row*0x973ab127u+k*0x9e3779b9u);
    return uint16_t(((x>>16)&0x8000u)|((126u+(x%2))<<7)|((x>>8)&127));
}
BOTH inline uint16_t w_bits(uint64_t i,unsigned h,bool tiny) {
    if(tiny) {
        const unsigned k=unsigned(i%h),n=unsigned(i/h);
        // Values exactly representable as BF16, using power-of-two scaling.
        const int q=k==0?int(n)-9:k==1?2*(int(n%3)-1):0;
        if(!q)return 0;
        unsigned a=unsigned(q<0?-q:q),e=0;while((1u<<(e+1))<=a)++e;
        return uint16_t((q<0?0x8000:0)|((e+123)<<7)|((a-(1u<<e))<<(7-e)));
    }
    const uint32_t x=mix(uint32_t(i)^mix(uint32_t(i>>32)+0x28a7134bu));
    return uint16_t(((x>>16)&0x8000u)|((117u+(x%6))<<7)|((x>>8)&127));
}
inline float decode(uint16_t x) {
    uint32_t bits=uint32_t(x)<<16;float f;std::memcpy(&f,&bits,4);return f;
}
inline void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
inline double oracle(unsigned row,unsigned n,unsigned h,bool tiny) {
    double sum=0;for(unsigned k=0;k<h;++k)
        sum+=double(decode(a_bits(row,k,tiny)))*double(decode(w_bits(uint64_t(n)*h+k,h,tiny)));
    return sum;
}
