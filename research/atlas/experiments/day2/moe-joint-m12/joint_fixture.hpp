// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "fixture.hpp"
struct CaseRoutes {
    unsigned rows;
    std::vector<unsigned>ids;
    std::vector<float>weights;
    std::vector<uint16_t>input;
    explicit CaseRoutes(const Routes&r):rows(M),ids(r.ids.begin(),r.ids.end()),weights(r.weights.begin(),r.weights.end()),input(r.input.begin(),r.input.end()){}
    explicit CaseRoutes(const std::vector<CaseRoutes>&owners):rows(0){
        need(owners.size()==4,"exactly four owners");
        for(const auto&r:owners){need(r.rows==3,"owner K3");rows+=r.rows;ids.insert(ids.end(),r.ids.begin(),r.ids.end());weights.insert(weights.end(),r.weights.begin(),r.weights.end());input.insert(input.end(),r.input.begin(),r.input.end());}
    }
};
inline std::vector<CaseRoutes> owner_routes(bool skew,bool shared=false){
    std::vector<CaseRoutes>v;constexpr unsigned shared_ids[8]={0,143,144,287,1,142,145,286};
    for(unsigned o=0;o<4;++o){v.emplace_back(Routes(skew,o));if(shared)for(unsigned t=0;t<3;++t)for(unsigned j=0;j<8;++j)v.back().ids[t*8+j]=shared_ids[j];}
    return v;
}
inline size_t work_bytes(unsigned rows){return 16+size_t(rows)*TOP*16*8;}
inline bool work_admitted(unsigned rows,size_t bytes){return (rows==3||rows==12)&&bytes>=work_bytes(rows);}
inline void check_routes(const CaseRoutes&r){
    need((r.rows==3||r.rows==12)&&r.ids.size()==r.rows*8&&r.weights.size()==r.ids.size()&&r.input.size()==r.rows*H,"route geometry");
    for(unsigned t=0;t<r.rows;++t){std::array<bool,E>seen{};for(unsigned j=0;j<8;++j){unsigned e=r.ids[t*8+j];need(e<E&&!seen[e],"unique legal top8");seen[e]=true;}}
}
inline unsigned checked_rows(const CaseRoutes&r,unsigned rank){check_routes(r);need(rank<2,"rank bound before allocation");return r.rows;}
inline std::vector<unsigned> expected_work(const CaseRoutes&r,unsigned rank){
    std::array<unsigned,E>counts{};for(auto e:r.ids)++counts[e];std::vector<unsigned>v;
    for(unsigned e=rank*144;e<(rank+1)*144;++e){need(counts[e]<=r.rows&&counts[e]<=12,"expert row bound");
        if(counts[e])for(unsigned nt=0;nt<16;++nt){v.push_back(e);v.push_back(nt);}}
    return v;
}
inline void check_work(const std::vector<unsigned>&work,const CaseRoutes&r,unsigned rank){
    auto expected=expected_work(r,rank);need(work.size()*4==work_bytes(r.rows)&&work[0]==expected.size()/2,"work count/extent");
    for(unsigned i=1;i<4;++i)need(work[i]==0xa5a5a5a5,"work header poison");
    for(size_t i=0;i<expected.size();++i)need(work[i+4]==expected[i],"work expert/tile coverage");
    for(size_t i=expected.size()+4;i<work.size();++i)need(work[i]==0xa5a5a5a5,"work tail poison");
}
inline void joint_host_tests(){
    unsigned checks=0;
    for(bool skew:{false,true})for(bool shared:{false,true}){
        auto owners=owner_routes(skew,shared);CaseRoutes joint(owners);check_routes(joint);
        std::array<unsigned,E>counts{};for(auto e:joint.ids)++counts[e];
        need(*std::max_element(counts.begin(),counts.end())==(shared?12u:(skew?3u:1u)),"declared joint occupancy");
        for(unsigned o=0;o<4;++o){check_routes(owners[o]);
            for(unsigned t=0;t<3;++t){unsigned global=o*3+t;need(global/3==o&&global%3==t,"owner flatten inverse");
                for(unsigned j=0;j<8;++j)need(joint.ids[global*8+j]==owners[o].ids[t*8+j]&&joint.weights[global*8+j]==owners[o].weights[t*8+j],"metadata flatten bytes");
                for(unsigned k=0;k<H;++k)need(joint.input[size_t(global)*H+k]==owners[o].input[size_t(t)*H+k],"input flatten bytes");}
            for(unsigned rank=0;rank<2;++rank){auto expected=expected_work(owners[o],rank);std::vector<unsigned>w(work_bytes(3)/4,0xa5a5a5a5);w[0]=expected.size()/2;std::copy(expected.begin(),expected.end(),w.begin()+4);check_work(w,owners[o],rank);++checks;}}
        for(unsigned rank=0;rank<2;++rank){auto expected=expected_work(joint,rank);std::vector<unsigned>w(work_bytes(12)/4,0xa5a5a5a5);w[0]=expected.size()/2;std::copy(expected.begin(),expected.end(),w.begin()+4);check_work(w,joint,rank);++checks;}
    }
    need(work_bytes(3)==3088&&work_bytes(12)==12304&&work_bytes(12)<16384,"work byte capacity");
    need(work_admitted(12,12304)&&!work_admitted(12,12303)&&!work_admitted(96,12304),"short/invalid capacity rejection");
    std::printf("PASS joint host: %u work maps, owner flatten, max12/shared,12303/12304 capacity\n",checks);
}
