// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Independent synthetic merge tests. No attention kernel or model inputs involved.
static void merge_checks(){
    constexpr unsigned S=16;const size_t plane=HEADS*DIM;
    Guarded po(S*plane*4),pl(S*HEADS*4),out(plane*2),lse(HEADS*4);
    std::vector<float>o(S*plane,NAN),l(S*HEADS,-INFINITY);
    auto reset=[&]{std::fill(o.begin(),o.end(),NAN);std::fill(l.begin(),l.end(),-INFINITY);};
    auto active=[&](unsigned s,float ll,float value){for(unsigned h=0;h<HEADS;++h)l[s*HEADS+h]=ll;
        for(size_t d=0;d<plane;++d)o[s*plane+d]=value;};
    for(unsigned test=0;test<9;++test){reset();double expected=0,expected_lse=-INFINITY;bool bad_lse=false;
        if(test==1){active(13,2,.5f);expected=.5;expected_lse=2;}
        if(test==2){active(0,float(std::log(2.)),1);active(7,float(std::log(3.)),5);double a=std::exp(double(l[0])),b=std::exp(double(l[7*HEADS]));expected=(a+5*b)/(a+b);expected_lse=std::log(a+b);}
        if(test==3){active(2,80,-.5f);active(11,80,.5f);expected=0;expected_lse=80+std::log(2.);}
        if(test==4){active(4,NAN,1);bad_lse=true;expected=NAN;expected_lse=NAN;}
        if(test==5){active(4,INFINITY,1);bad_lse=true;expected=NAN;expected_lse=NAN;}
        if(test==6){active(4,0,NAN);expected=NAN;expected_lse=0;}
        if(test==7){active(4,0,INFINITY);expected=INFINITY;expected_lse=0;}
        if(test==8){active(4,0,-INFINITY);expected=-INFINITY;expected_lse=0;}
        po.upload(o.data());pl.upload(l.data());out.poison();lse.poison();
        CUDA(static_cast<cudaError_t>(merge_attention_parts(po.as<float>(),pl.as<float>(),out.data(),lse.as<float>(),1,HEADS,DIM,S,nullptr)));
        CUDA(cudaDeviceSynchronize());auto y=out.output();auto z=lse.floats();
        for(auto v:y){float f=from_bf16(v);if(std::isnan(expected))require(std::isnan(f),"merge NaN propagation");
            else if(std::isinf(expected))require(f==expected,"merge active infinity propagation");
            else require(v==to_bf16(float(expected)),"merge independent weighted output bits");}
        for(auto f:z){if(bad_lse)require(std::isnan(f),"merge nonfinite LSE poison");else check_lse(f,expected_lse);}
        po.unchanged(o.data());pl.unchanged(l.data());
        std::printf("{\"stage\":\"merge_special\",\"case\":%u,\"pass\":true}\n",test);
    }
    out.poison();lse.poison();const auto before=out.read(),before_lse=lse.read();
    auto rejected=[&](const float*a,const float*b,void*c,float*d,unsigned rows,unsigned heads,unsigned dim,unsigned splits){
        int status=merge_attention_parts(a,b,c,d,rows,heads,dim,splits,nullptr);
        require(status==int(cudaErrorInvalidValue),"invalid merge must reject before launch");};
    auto*a=po.as<float>();auto*b=pl.as<float>();auto*c=out.data();auto*d=lse.as<float>();
    rejected(nullptr,b,c,d,1,32,512,8);rejected(a,nullptr,c,d,1,32,512,8);rejected(a,b,nullptr,d,1,32,512,8);rejected(a,b,c,nullptr,1,32,512,8);
    for(unsigned rows:{0u,5u})rejected(a,b,c,d,rows,32,512,8);
    rejected(a,b,c,d,1,31,512,8);rejected(a,b,c,d,1,32,511,8);
    for(unsigned splits:{0u,17u})rejected(a,b,c,d,1,32,512,splits);
    rejected(reinterpret_cast<float*>(reinterpret_cast<char*>(a)+1),b,c,d,1,32,512,8);
    rejected(a,reinterpret_cast<float*>(reinterpret_cast<char*>(b)+1),c,d,1,32,512,8);
    rejected(a,b,reinterpret_cast<char*>(c)+1,d,1,32,512,8);
    rejected(a,b,c,reinterpret_cast<float*>(reinterpret_cast<char*>(d)+1),1,32,512,8);
    CUDA(cudaDeviceSynchronize());require(out.read()==before&&lse.read()==before_lse,"rejected merge wrote destinations");
    std::puts("{\"stage\":\"merge_rejections\",\"cases\":14,\"destinations_untouched\":true}");
}
