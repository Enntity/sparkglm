// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "fixture.hpp"
struct Snapshot {
    unsigned inter,rank;bool tp;
    std::vector<unsigned>top;
    std::vector<float>weights;
    std::vector<int>tokens,experts,offsets,reverse;
    std::vector<uint16_t>input,gate,up,down,out;
    std::vector<unsigned char>a,aq,packed;
};
inline double activation(const std::vector<unsigned char>&p,size_t base,size_t scale_base,unsigned k) {
    return e2((p[base+k/2]>>((k&1)*4))&15)*e4(p[scale_base+k/16]);
}
inline void finite16(uint16_t b){need((b&0x7f80)!=0x7f80,"finite BF16 output");}
inline void dot_check(double dot,double sumabs,unsigned k,float actual,unsigned e,unsigned c,const char* projection) {
    // One full BF16 ULP plus gamma(K) FP32 accumulation, and scale multiplication.
    double gamma=(double(k)+2)*std::ldexp(1.,-24)/(1-(double(k)+2)*std::ldexp(1.,-24));
    double ulp=dot==0?std::ldexp(1.,-133):std::ldexp(1.,std::ilogb(std::abs(dot))-7);
    double limit=ulp+gamma*sumabs+1e-12;
    if(!std::isfinite(actual)||std::abs(actual-dot)>limit){std::fprintf(stderr,"FAIL FP64 %s expert=%u col=%u ref=%.12g actual=%.12g abs=%.9g bound=%.9g\n",projection,e,c,dot,actual,std::abs(actual-dot),limit);std::exit(2);}
}
inline void check_snapshot(const Snapshot&s,const Routes&r) {
    need(s.top==std::vector<unsigned>(r.ids.begin(),r.ids.end()),"topk IDs immutable");
    need(s.weights==std::vector<float>(r.weights.begin(),r.weights.end()),"routing weights immutable");
    need(s.input==std::vector<uint16_t>(r.input.begin(),r.input.end()),"input immutable");
    need(s.offsets.size()==E+1&&s.offsets[0]==0&&s.offsets[E]==int(R),"sort offset extent");
    std::array<bool,R> seen{};unsigned dots=0;
    for(unsigned slot=0;slot<R;++slot){unsigned t=slot/TOP,e=r.ids[slot];int pos=s.reverse[slot];
        need(pos>=0&&pos<int(R)&&!seen[pos],"permutation bijection");seen[pos]=true;
        need(s.tokens[pos]==int(t)&&s.experts[pos]==int(e)&&pos>=s.offsets[e]&&pos<s.offsets[e+1],"sort mapping");
        bool owned=s.tp||e/144==s.rank;
        for(unsigned c=0;c<s.inter;++c){auto g=s.gate[size_t(pos)*s.inter+c],u=s.up[size_t(pos)*s.inter+c];
            if(owned){finite16(g);finite16(u);}else need(g==0xa5a5&&u==0xa5a5,"remote gate/up poison");}
        for(unsigned c=0;c<H;++c){auto v=s.down[size_t(pos)*H+c];if(owned)finite16(v);else need(v==0xa5a5,"remote down poison");}
        if(!owned)continue;
        for(unsigned proj=0;proj<2;++proj)for(unsigned c:{0u,31u,s.inter-1}) {
            double d=0,a=0;unsigned gc=c+(s.tp?s.rank*I/2:0);
            for(unsigned k=0;k<H;++k){double av=e2((s.a[size_t(t)*H/2+k/2]>>((k&1)*4))&15)*e4(s.aq[size_t(t)*H/16+k/16]);
                double v=av*w_value(proj,e,gc,k);d+=v;a+=std::abs(v);}
            dot_check(d,a,H,bf((proj?s.up:s.gate)[size_t(pos)*s.inter+c]),e,gc,proj?"up":"gate");++dots;
        }
        for(unsigned c:{0u,31u,4095u}){double d=0,a=0;
            for(unsigned k=0;k<s.inter;++k){double av=activation(s.packed,size_t(pos)*s.inter/2,size_t(R)*s.inter/2+size_t(pos)*s.inter/16,k);
                double v=av*w_value(2,e,c,k+(s.tp?s.rank*I/2:0));d+=v;a+=std::abs(v);}
            dot_check(d,a,s.inter,bf(s.down[size_t(pos)*H+c]),e,c,"down partial");++dots;}
    }
    for(unsigned t=0;t<M;++t)for(unsigned c=0;c<H;++c){float acc=0;
        for(unsigned j=0;j<TOP;++j){unsigned slot=t*TOP+j,e=s.top[slot];if(s.tp||e/144==s.rank){float product=s.weights[slot]*bf(s.down[size_t(s.reverse[slot])*H+c]);acc=acc+product;}}
        need(round_bf(acc)==s.out[size_t(t)*H+c],"independent weighted unpermute exact bits");finite16(s.out[size_t(t)*H+c]);}
    std::printf("PASS snapshot tp=%u rank=%u independent_fp64_dots=%u\n",s.tp,s.rank,dots);
}
// Both arms use EP2 with identical full expert matrices. No TP sharding tolerance.
inline void compare_compact(const std::array<Snapshot,4>&s,const Routes&r,const std::vector<uint16_t>&dense,const std::vector<uint16_t>&compact) {
    for(auto&v:s)check_snapshot(v,r);
    size_t exact=0;
    for(unsigned rank=0;rank<2;++rank){const auto&a=s[rank];const auto&b=s[2+rank];
        need(!a.tp&&!b.tp&&a.rank==rank&&b.rank==rank&&a.inter==I&&b.inter==I,"identical EP geometry");
        need(a.top==b.top&&a.weights==b.weights&&a.tokens==b.tokens&&a.experts==b.experts&&a.offsets==b.offsets&&a.reverse==b.reverse,"all routing metadata identical");
        need(a.input==b.input&&a.a==b.a&&a.aq==b.aq&&a.packed==b.packed,"all input/quantized activation bytes identical");
        need(a.gate==b.gate&&a.up==b.up&&a.down==b.down&&a.out==b.out,"all owned BF16 bits and remote poison identical");
        exact+=(a.gate.size()+a.up.size()+a.down.size()+a.out.size())*2+a.a.size()+a.aq.size()+a.packed.size();
    }
    need(dense.size()==M*H&&compact.size()==dense.size(),"pair sum extent");
    bool nonzero=false;
    for(size_t i=0;i<dense.size();++i){need(round_bf(bf(s[0].out[i])+bf(s[1].out[i]))==dense[i],"dense independent pair-sum bits");
        need(round_bf(bf(s[2].out[i])+bf(s[3].out[i]))==compact[i],"compact independent pair-sum bits");
        need(dense[i]==compact[i],"paired output exact bits");finite16(dense[i]);nonzero|=(dense[i]&0x7fff)!=0;}
    need(nonzero,"nontrivial routed result");
    std::printf("PASS COMPACT_ORACLE exact_pipeline_bytes=%zu exact_pair_elements=%zu\n",exact,dense.size());
}
