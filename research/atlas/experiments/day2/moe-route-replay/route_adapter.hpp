// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "joint_fixture.hpp"
struct ReplayLayer {
    unsigned round,layer;
    std::array<unsigned,4> slots,bases;
    std::array<uint64_t,4> generations;
    std::array<std::array<unsigned,24>,4> ids,weight_bits;
};
inline std::vector<CaseRoutes> routes_from_capture(const ReplayLayer&c){
    need((c.round==1||c.round==8)&&c.layer>=3&&c.layer<45,"fixed replay round/layer");
    std::vector<CaseRoutes> v;
    for(unsigned o=0;o<4;++o){
        need(c.slots[o]<4&&c.generations[o]>0&&c.bases[o]>=2048&&c.bases[o]<=32765,"captured owner identity/base");
        for(unsigned j=0;j<o;++j)need(c.slots[j]!=c.slots[o],"distinct captured slots");
        // Synthetic input/weight recipe is unchanged; only IDs and route weight bits change.
        v.emplace_back(Routes(false,o));
        for(unsigned i=0;i<24;++i){float w=f32(c.weight_bits[o][i]);need(std::isfinite(w)&&w>=0,"finite nonnegative captured weight");
            v.back().ids[i]=c.ids[o][i];v.back().weights[i]=w;need(bits32(v.back().weights[i])==c.weight_bits[o][i],"exact FP32 route weight bits");}
        check_routes(v.back());
    }
    return v;
}
