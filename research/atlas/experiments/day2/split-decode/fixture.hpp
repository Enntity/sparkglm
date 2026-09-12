// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr unsigned HEADS=32, DIM=512, BLOCK=16;
constexpr float SCALE=0.0625f;
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline uint16_t to_bf16(float x) {
    uint32_t bits; std::memcpy(&bits,&x,4);
    bits += 0x7fff + ((bits>>16)&1);
    return uint16_t(bits>>16);
}
inline float from_bf16(uint16_t x) {
    uint32_t bits=uint32_t(x)<<16; float value;
    std::memcpy(&value,&bits,4); return value;
}
enum class Kind { Random, Singleton, Constant, Empty, AllMasked };
struct Spec {
    const char* name;
    unsigned rows, width, history;
    Kind kind;
    unsigned tail;
    bool holes;
};
struct Fixture {
    Spec spec;
    unsigned physical_blocks;
    std::vector<uint16_t> query, kv;
    std::vector<int32_t> ids;
    std::vector<uint32_t> table;
    size_t elements() const { return size_t(spec.rows)*HEADS*DIM; }
    size_t device_bytes() const {
        return kv.size()*2 + elements()*6 + std::max(size_t(1),ids.size())*4
            + table.size()*4 + 6*512;
    }
};
inline bool exact_kind(Kind k) { return k != Kind::Random; }
inline float constant_value(unsigned d) {
    constexpr float v[4]={-0.5f,-0.25f,0.25f,0.5f}; return v[d%4];
}
inline Fixture make_fixture(Spec s) {
    require((s.rows==1 || s.rows==3) && s.width<=2051 && s.history>=64
        && s.history<=32768 && s.history%BLOCK==0, "fixture construction shape");
    const unsigned logical=s.history/BLOCK;
    Fixture f{s, logical+2, {}, {}, {}, {}};
    f.query.resize(f.elements());
    f.kv.assign(size_t(f.physical_blocks)*BLOCK*DIM,0xffff);
    f.ids.assign(size_t(s.rows)*s.width,-1);
    f.table.resize(logical);
    std::mt19937 random(92731+s.rows*17+s.width*7+s.history);
    auto bounded=[&] { return to_bf16(float(int(random()%257)-128)/128.0f); };
    for (auto& x:f.query) x=bounded();
    for (unsigned i=0;i<logical;++i) f.table[i]=1+(i*2053+17)%logical;
    for (unsigned physical=1;physical+1<f.physical_blocks;++physical)
        for (unsigned t=0;t<BLOCK;++t)
            for (unsigned d=0;d<DIM;++d)
                f.kv[(size_t(physical)*BLOCK+t)*DIM+d]=s.kind==Kind::Constant
                    ? to_bf16(constant_value(d)) : bounded();
    for (unsigned r=0;r<s.rows;++r) {
        if (s.kind==Kind::Singleton) {
            if (s.width) f.ids[size_t(r)*s.width+s.width-1]=int((7+r*11)%s.history);
            continue;
        }
        if (s.kind==Kind::Empty || s.kind==Kind::AllMasked) continue;
        const unsigned length=s.history-8+s.tail+r;
        const unsigned pools=length/4, tail=length%4;
        for (unsigned i=0;i<s.width;++i) {
            int token;
            if (s.width<2048) token=int((i*19+r*7)%s.history);
            else if (i<2048) token=int((((i/4)*11+r*37)%pools)*4+i%4);
            else token=i-2048<tail ? int(pools*4+i-2048) : -1;
            if (s.holes && (i%13==0 || (i>=32 && i<64))) token=-1;
            f.ids[size_t(r)*s.width+i]=token;
        }
    }
    return f;
}
inline void validate(const Fixture& f) {
    const auto& s=f.spec;
    require((s.rows==1 || s.rows==3) && s.width<=2051 && s.history>=64
        && s.history<=32768 && s.history%BLOCK==0 && s.tail<=3, "host shape");
    require(f.physical_blocks==s.history/BLOCK+2, "physical block capacity");
    require(f.query.size()==f.elements() && f.kv.size()==size_t(f.physical_blocks)*BLOCK*DIM
        && f.ids.size()==size_t(s.rows)*s.width && f.table.size()==s.history/BLOCK,
        "host payload capacity");
    std::vector<bool> seen(f.physical_blocks);
    for (const auto block:f.table) {
        require(block>0 && block+1<f.physical_blocks && !seen[block], "physical table bounds/alias");
        seen[block]=true;
    }
    for (const auto token:f.ids)
        require(token==-1 || (token>=0 && unsigned(token)<s.history), "selected ID bounds");
    for (const auto x:f.query) require(std::isfinite(from_bf16(x)), "query finite");
    for (unsigned p=1;p+1<f.physical_blocks;++p)
        for (size_t i=size_t(p)*BLOCK*DIM;i<size_t(p+1)*BLOCK*DIM;++i)
            require(std::isfinite(from_bf16(f.kv[i])), "cache finite");
    require(f.device_bytes()<128u*1024u*1024u, "device memory cap");
}
inline size_t kv_position(const Fixture& f,int token) {
    return (size_t(f.table[unsigned(token)/BLOCK])*BLOCK+unsigned(token)%BLOCK)*DIM;
}
inline uint16_t exact_output(const Fixture& f,size_t element) {
    if (f.spec.kind==Kind::Empty || f.spec.kind==Kind::AllMasked) return 0;
    if (f.spec.kind==Kind::Constant) return to_bf16(constant_value(unsigned(element%DIM)));
    const unsigned row=unsigned(element/(HEADS*DIM));
    require(f.spec.kind==Kind::Singleton && f.spec.width>0,"exact output kind");
    const int token=f.ids[size_t(row)*f.spec.width+f.spec.width-1];
    return f.kv[kv_position(f,token)+element%DIM];
}
inline std::vector<double> oracle(const Fixture& f) {
    validate(f);
    std::vector<double> out(f.elements(),0.0), scores(f.spec.width);
    for (unsigned r=0;r<f.spec.rows;++r) for (unsigned h=0;h<HEADS;++h) {
        const size_t q=(size_t(r)*HEADS+h)*DIM;
        double maximum=-INFINITY;
        for (unsigned i=0;i<f.spec.width;++i) {
            const int token=f.ids[size_t(r)*f.spec.width+i];
            double dot=-INFINITY;
            if (token>=0) {
                dot=0; const size_t k=kv_position(f,token);
                for (unsigned d=0;d<DIM;++d)
                    dot+=double(from_bf16(f.query[q+d]))*double(from_bf16(f.kv[k+d]));
                dot*=double(SCALE);
            }
            scores[i]=dot; maximum=std::max(maximum,dot);
        }
        double denominator=0;
        for (auto& score:scores) { score=std::isfinite(score)?std::exp(score-maximum):0; denominator+=score; }
        if (!denominator) continue;
        for (unsigned i=0;i<f.spec.width;++i) {
            const int token=f.ids[size_t(r)*f.spec.width+i];
            if (token<0) continue;
            const size_t k=kv_position(f,token);
            for (unsigned d=0;d<DIM;++d) out[q+d]+=scores[i]*double(from_bf16(f.kv[k+d]));
        }
        for (unsigned d=0;d<DIM;++d) out[q+d]/=denominator;
    }
    return out;
}
inline std::vector<Spec> correctness_specs() {
    return {
        {"singleton",1,1,64,Kind::Singleton,0,false},
        {"singleton_after_empty_tiles",3,2051,64,Kind::Singleton,0,false},
        {"constant_values",3,2051,64,Kind::Constant,3,false},
        {"empty",3,0,64,Kind::Empty,0,false},
        {"all_invalid",3,2051,64,Kind::AllMasked,0,false},
        {"tiny_holes",3,19,64,Kind::Random,0,true},
        {"m1_w2048",1,2048,16384,Kind::Random,0,false},
        {"m1_w2049",1,2049,16384,Kind::Random,1,false},
        {"m1_w2050",1,2050,16384,Kind::Random,2,false},
        {"m1_w2051",1,2051,16384,Kind::Random,3,false},
        {"m3_w2051",3,2051,16384,Kind::Random,3,false},
        {"m1_empty_tail",1,2051,16384,Kind::Random,0,false},
        {"m3_holes",3,2051,16384,Kind::Random,0,true},
        {"m1_32k",1,2051,32768,Kind::Random,3,false},
    };
}
