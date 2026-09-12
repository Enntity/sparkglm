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
inline void compare(const std::array<Snapshot,4>&s,const Routes&r,const std::vector<uint16_t>&ep,const std::vector<uint16_t>&tp) {
    for(auto&v:s)check_snapshot(v,r);
    size_t checked=0;
    for(unsigned slot=0;slot<R;++slot){unsigned er=r.ids[slot]/144;auto&b=s[er];unsigned br=unsigned(b.reverse[slot]);
        for(unsigned rank=0;rank<2;++rank){auto&v=s[2+rank];unsigned tr=unsigned(v.reverse[slot]);
            for(unsigned c=0;c<I/2;++c){size_t bi=size_t(br)*I+rank*I/2+c,ti=size_t(tr)*I/2+c;
                need(b.gate[bi]==v.gate[ti]&&b.up[bi]==v.up[ti],"all gate/up shard BF16 bits");checked+=4;}
            for(unsigned div:{2u,16u}){size_t bo=(div==16?size_t(R)*I/2:0)+size_t(br)*I/div+rank*I/(2*div);
                size_t to=(div==16?size_t(R)*I/4:0)+size_t(tr)*I/(2*div);
                for(unsigned c=0;c<I/(2*div);++c){need(b.packed[bo+c]==v.packed[to+c],"all SiLU FP4/scales shard bytes");++checked;}}
        }
    }
    double norm=0,err=0,maxerr=0,maxref=0;
    for(size_t i=0;i<ep.size();++i){need(round_bf(bf(s[0].out[i])+bf(s[1].out[i]))==ep[i],"EP independent pair-sum bits");
        need(round_bf(bf(s[2].out[i])+bf(s[3].out[i]))==tp[i],"TP independent pair-sum bits");
        double b=bf(ep[i]),d=bf(tp[i])-b;norm+=b*b;err+=d*d;maxerr=std::max(maxerr,std::abs(d));maxref=std::max(maxref,std::abs(b));}
    need(norm>0&&maxref>0,"nontrivial routed result");double rel=std::sqrt(err/norm);
    std::printf("NUMERICS exact_shard_bytes=%zu final_rel_l2=%.9g final_max_abs=%.9g max_ref=%.9g\n",checked,rel,maxerr,maxref);
    need(rel<=.01&&maxerr<=.02*maxref,"predeclared EP/TP routed numerical gate");
}
