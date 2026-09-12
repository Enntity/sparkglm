// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
struct Error { double maximum=0, relative=0; };
static Error compare(const Fixture& f,const std::vector<uint16_t>& got,const std::vector<double>& ref) {
    require(got.size()==ref.size(),"output element count");
    Error e; double squares=0,norm=0;
    for(size_t i=0;i<got.size();++i) {
        const double value=from_bf16(got[i]);
        require(std::isfinite(value) && std::isfinite(ref[i]),"nonfinite or unwritten output");
        const double delta=value-ref[i];
        e.maximum=std::max(e.maximum,std::abs(delta)); squares+=delta*delta; norm+=ref[i]*ref[i];
        if(exact_kind(f.spec.kind)) require(got[i]==exact_output(f,i),"exact fixture mismatch");
    }
    e.relative=std::sqrt(squares/std::max(norm,1e-30));
    require(e.maximum<=0.03 && e.relative<=0.01,"FP64 tolerance failed: maxabs0.03/relativeL2 0.01");
    return e;
}
static void samples(const std::vector<float>& v) {
    std::putchar('['); for(size_t i=0;i<v.size();++i) std::printf("%s%.9g",i?",":"",v[i]); std::putchar(']');
}
