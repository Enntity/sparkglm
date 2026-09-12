// SPDX-License-Identifier: AGPL-3.0-only
#include "fixture.hpp"
#include <cstdio>
int main(){
    for(unsigned r=0;r<M;++r)for(unsigned n=0;n<20;++n){
        const double expected=(double(r)+1)*(int(n)-9)/16.0-(int(n%3)-1)/16.0;
        require(oracle(r,n,16,true)==expected,"tiny analytical oracle mismatch");
    }
    for(unsigned r=0;r<M;++r)for(unsigned k=0;k<H;++k)
        require(std::isfinite(decode(a_bits(r,k,false))),"activation finite");
    for(uint64_t i=0;i<65536;++i)
        require(std::isfinite(decode(w_bits(i*9679,H,false))),"weight finite");
    require(V%4==0 && H%8==0,"production vector/barrier geometry");
    std::puts("PASS: 60 analytical tiny dots, all activation rows, sampled weight generator, geometry");
}
