// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include "fixture.hpp"
#include <array>
inline std::pair<unsigned,unsigned> interval(unsigned width,unsigned split,unsigned splits){
    unsigned tiles=(width+31)/32;return {tiles*split/splits*32,std::min(width,tiles*(split+1)/splits*32)};
}
inline double logsum(const std::vector<double>&v,size_t start,size_t count){
    double mx=-INFINITY;for(size_t i=0;i<count;++i)mx=std::max(mx,v[start+i]);
    if(mx==-INFINITY)return mx;
    double sum=0;for(size_t i=0;i<count;++i)if(v[start+i]!=-INFINITY)sum+=std::exp(v[start+i]-mx);
    return mx+std::log(sum);
}
inline std::vector<double> reference_scores(const Fixture&f){
    std::vector<double>scores(size_t(f.spec.rows)*HEADS*f.spec.width,-INFINITY);
    for(unsigned r=0;r<f.spec.rows;++r)for(unsigned h=0;h<HEADS;++h)for(unsigned i=0;i<f.spec.width;++i){
        int token=f.ids[size_t(r)*f.spec.width+i];if(token<0)continue;double dot=0;
        size_t q=(size_t(r)*HEADS+h)*DIM,k=kv_position(f,token);
        for(unsigned d=0;d<DIM;++d)dot+=double(from_bf16(f.query[q+d]))*from_bf16(f.kv[k+d]);
        scores[(size_t(r)*HEADS+h)*f.spec.width+i]=dot*SCALE;
    }
    return scores;
}
inline void check_lse(float actual,double expected){
    if(expected==-INFINITY)require(actual==-INFINITY,"empty LSE must be negative infinity");
    else require(std::isfinite(actual)&&std::abs(double(actual)-expected)<=2e-4,"natural FP64 LSE tolerance2e-4");
}
inline void host_tests(){
    for(unsigned width=0;width<=2051;++width)for(unsigned splits:{1u,2u,4u,8u,16u}){
        std::vector<unsigned>coverage(width);unsigned previous=0;
        for(unsigned s=0;s<splits;++s){auto[lo,hi]=interval(width,s,splits);require(lo<=hi&&hi<=width&&lo==previous,"contiguous partition bounds");
            require(lo%32==0&& (hi==width||hi%32==0),"partition tile alignment");for(unsigned i=lo;i<hi;++i)++coverage[i];previous=hi;}
        require(previous==width,"partition reaches tail");for(unsigned c:coverage)require(c==1,"partition exact coverage");
    }
    require(logsum({},0,0)==-INFINITY,"empty natural LSE");
    require(std::abs(logsum({0,0,0,0},0,4)-std::log(4.))<1e-14,"natural not log2 convention");
    require(std::abs(logsum({std::log(2.),std::log(3.)},0,2)-std::log(5.))<1e-14,"weighted natural LSE");
    for(const auto&s:correctness_specs())validate(make_fixture(s));
}
