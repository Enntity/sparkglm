// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "fixture.hpp"
constexpr size_t work_bytes(unsigned expanded,unsigned nt){return 16+size_t(expanded)*nt*8;}
constexpr size_t WORK_BYTES=work_bytes(R,H/128);
static_assert(WORK_BYTES==6160 && WORK_BYTES<=16384,"fits minimum production router scratch");
inline bool work_admitted(size_t bytes){return bytes>=WORK_BYTES;}
// Independent enumeration from original top-k IDs (not GPU offsets or builder results).
inline std::vector<unsigned> expected_work(const Routes&r,unsigned rank,unsigned nt){
    std::array<unsigned,E> counts{};for(auto e:r.ids){need(e<E,"route expert range");++counts[e];}
    std::vector<unsigned> v;
    for(unsigned e=rank*144;e<(rank+1)*144;++e)for(unsigned mt=0;mt<(counts[e]+63)/64;++mt)
        for(unsigned n=0;n<nt;++n){v.push_back(e);v.push_back((mt<<6)|n);}
    return v;
}
inline bool valid_work(const std::vector<unsigned>&w,const Routes&r,unsigned rank,unsigned nt){
    auto e=expected_work(r,rank,nt);
    if(w.size()!=WORK_BYTES/4 || w[0]!=e.size()/2 || e.size()+4>w.size())return false;
    for(unsigned i=1;i<4;++i)if(w[i]!=0xa5a5a5a5)return false;
    for(size_t i=0;i<e.size();++i)if(w[i+4]!=e[i])return false;
    for(size_t i=e.size()+4;i<w.size();++i)if(w[i]!=0xa5a5a5a5)return false;
    return true;
}
inline void check_worklist(const std::vector<unsigned>&w,const Routes&r,unsigned rank,unsigned nt){
    need(valid_work(w,r,rank,nt),"exact worklist: count, owned expert/tile coverage, header/tail poison");
}
inline void worklist_host_tests(){
    need(work_admitted(6160)&&work_admitted(16384)&&!work_admitted(6159),"capacity rejects one byte short");
    unsigned cases=0;
    for(bool skew:{false,true})for(unsigned owner=0;owner<4;++owner)for(unsigned rank=0;rank<2;++rank)for(unsigned nt:{16u,32u}){
        Routes r(skew,owner);auto e=expected_work(r,rank,nt);need(e.size()/2<=R*nt,"worklist worst case bound");
        std::vector<unsigned>w(WORK_BYTES/4,0xa5a5a5a5);w[0]=unsigned(e.size()/2);std::copy(e.begin(),e.end(),w.begin()+4);
        check_worklist(w,r,rank,nt);w[4]^=1;need(!valid_work(w,r,rank,nt),"wrong expert rejected");++cases;
    }
    Routes all(false,0);for(unsigned i=0;i<R;++i)all.ids[i]=i;
    need(expected_work(all,0,32).size()*4+16==WORK_BYTES,"24 owned unique experts reaches6160 bound");
    need(expected_work(all,1,32).empty(),"empty remote-only rank emits no work");
    std::printf("PASS host worklist: %u exact route/rank/tile cases, capacity6159/6160, max/empty coverage\n",cases);
}
