// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "joint_fixture.hpp"
struct Snapshot {
    unsigned rows,inter,rank;bool tp;
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
inline void check_snapshot(const Snapshot&s,const CaseRoutes&r) {
    const unsigned M=s.rows,R=M*TOP;need(M==r.rows,"snapshot row extent");
    need(s.top==std::vector<unsigned>(r.ids.begin(),r.ids.end()),"topk IDs immutable");
    need(s.weights==std::vector<float>(r.weights.begin(),r.weights.end()),"routing weights immutable");
    need(s.input==std::vector<uint16_t>(r.input.begin(),r.input.end()),"input immutable");
    need(s.offsets.size()==E+1&&s.offsets[0]==0&&s.offsets[E]==int(R),"sort offset extent");
    std::vector<bool> seen(R,false);unsigned dots=0;
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
inline void compare_joint(const std::array<std::array<Snapshot,2>,4>&serial,const std::array<Snapshot,2>&joint,const std::vector<CaseRoutes>&owners,const CaseRoutes&flat,const std::array<std::vector<uint16_t>,4>&serial_sum,const std::array<std::vector<uint16_t>,4>&joint_sum){
    size_t route_bytes=0,output_elements=0;
    for(unsigned rank=0;rank<2;++rank){check_snapshot(joint[rank],flat);const auto&b=joint[rank];need(b.rows==12&&!b.tp,"joint EP shape");
        for(unsigned o=0;o<4;++o){const auto&a=serial[o][rank];check_snapshot(a,owners[o]);need(a.rows==3&&!a.tp,"serial EP shape");
            for(unsigned t=0;t<3;++t){unsigned global=o*3+t;
                for(unsigned k=0;k<H/2;++k)need(a.a[size_t(t)*H/2+k]==b.a[size_t(global)*H/2+k],"all token FP4 bytes exact");
                for(unsigned k=0;k<H/16;++k)need(a.aq[size_t(t)*H/16+k]==b.aq[size_t(global)*H/16+k],"all token scale bytes exact");
                for(unsigned j=0;j<8;++j){unsigned slot=t*8+j,gslot=global*8+j,ar=a.reverse[slot],br=b.reverse[gslot];
                    need(a.experts[ar]==b.experts[br]&&a.tokens[ar]==int(t)&&b.tokens[br]==int(global),"canonical route identity");
                    for(unsigned c=0;c<I;++c){need(a.gate[size_t(ar)*I+c]==b.gate[size_t(br)*I+c],"gate bits/remote poison");need(a.up[size_t(ar)*I+c]==b.up[size_t(br)*I+c],"up bits/remote poison");route_bytes+=4;}
                    for(unsigned c=0;c<H;++c){need(a.down[size_t(ar)*H+c]==b.down[size_t(br)*H+c],"down bits/remote poison");route_bytes+=2;}
                    for(unsigned div:{2u,16u}){size_t abase=(div==16?24*I/2:0)+size_t(ar)*I/div,bbase=(div==16?96*I/2:0)+size_t(br)*I/div;
                        for(unsigned c=0;c<I/div;++c){need(a.packed[abase+c]==b.packed[bbase+c],"canonical SiLU FP4/scales exact");++route_bytes;}}
                }
                for(unsigned c=0;c<H;++c){need(a.out[size_t(t)*H+c]==b.out[size_t(global)*H+c],"owner-local unpermute exact");++output_elements;}
            }
        }
    }
    bool nonzero=false;
    for(unsigned o=0;o<4;++o){need(serial_sum[o].size()==3*H&&joint_sum[o].size()==3*H,"owner scatter extent");
        for(unsigned i=0;i<3*H;++i){need(serial_sum[o][i]==joint_sum[o][i],"owner pair/scatter exact");
            need(serial_sum[o][i]==round_bf(bf(serial[o][0].out[i])+bf(serial[o][1].out[i])),"serial independent pair bits");
            need(joint_sum[o][i]==round_bf(bf(joint[0].out[size_t(o)*3*H+i])+bf(joint[1].out[size_t(o)*3*H+i])),"joint independent pair bits");
            finite16(joint_sum[o][i]);nonzero|=(joint_sum[o][i]&0x7fff)!=0;}}
    need(nonzero,"nontrivial joint result");
    std::printf("PASS JOINT_ORACLE canonical_route_bytes=%zu local_output_elements=%zu owner_pair_elements=%u\n",route_bytes,output_elements,12*H);
}
