// SPDX-License-Identifier: AGPL-3.0-only
// Generic adapter-only host test; never emits or substitutes an actual capture header.
#include "route_adapter.hpp"
int main(){host_tests();joint_host_tests();ReplayLayer c{};c.round=1;c.layer=3;
    for(unsigned o=0;o<4;++o){c.slots[o]=(o+1)%4;c.generations[o]=c.slots[o]+1;c.bases[o]=2048+o;
        for(unsigned i=0;i<24;++i){c.ids[o][i]=(o*40+i)%288;c.weight_bits[o][i]=bits32(.5f);}}
    auto v=routes_from_capture(c);CaseRoutes flat(v);check_routes(flat);
    for(unsigned o=0;o<4;++o)for(unsigned i=0;i<24;++i){need(v[o].ids[i]==c.ids[o][i],"captured IDs copied exactly");
        need(bits32(v[o].weights[i])==c.weight_bits[o][i],"no weight normalization");}
    c.weight_bits[0][0]=0x80000000u;auto zero=routes_from_capture(c);need(bits32(zero[0].weights[0])==0x80000000u,"signed zero bits preserved");
    for(unsigned r=0;r<2;++r){auto work=expected_work(flat,r);need(work.size()/2<=96*16,"max96 route work capacity");}
    std::puts("PASS adapter host exact IDs/weights/owner-order/flat-M12; no actual route data supplied");}
