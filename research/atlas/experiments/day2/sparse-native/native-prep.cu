// SPDX-License-Identifier: AGPL-3.0-only
// Supervised DeepSeek mechanical kernel draft adapted in a separate header; original
// checked wrappers below. See NATIVE_PREP_PLAN.md and retained response receipt.
// nvcc -std=c++17 -O2 --fmad=false --prec-div=true --ftz=false \
//   -gencode=arch=compute_121a,code=sm_121a -shared -Xcompiler=-fPIC \
//   native-prep.cu -o libatlas_native_prep.so
#include "native-prep-reviewed.cuh"
#include <limits>

namespace {
bool overlap(const void* a, uint64_t na, const void* b, uint64_t nb) {
    uintptr_t x=reinterpret_cast<uintptr_t>(a), y=reinterpret_cast<uintptr_t>(b);
    if(na>UINTPTR_MAX-x || nb>UINTPTR_MAX-y) return true;
    return x<y+nb && y<x+na;
}
bool aligned(const void* p,uintptr_t n) {return reinterpret_cast<uintptr_t>(p)%n==0;}
}
extern "C" int atlas_native_prep_version() {return 1;}
// qbytes/outbytes are allocation capacities from the supplied base addresses.
extern "C" int atlas_native_prep_q(const void* q,void* out,uint32_t rows,
    uint64_t qbytes,uint64_t outbytes,void* stream) {
    if(!q||!out) return -1;
    if(rows==0||rows>32768) return -2;
    uint64_t ni=uint64_t(rows)*32*512*2, no=uint64_t(rows)*32*576*2;
    if(qbytes<ni||outbytes<no) return -3;
    if(!aligned(q,2)||!aligned(out,2)||overlap(q,ni,out,no)) return -4;
    prep_q<<<unsigned((no/2+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(q),static_cast<__nv_bfloat16*>(out),rows);
    return int(cudaGetLastError());
}
// paged[pblocks,16,512], bt[btcount]. Caller guarantees each used bt value is
// in [0,pblocks); no device-table inspection or synchronization in this wrapper.
extern "C" int atlas_native_prep_kv(const void* paged,uint32_t pblocks,
    const int32_t* bt,uint32_t btcount,void* out,uint64_t outbytes,
    uint32_t seqend,void* stream) {
    if(!paged||!bt||!out) return -1;
    if(seqend==0||seqend>32768||pblocks==0||pblocks>32768) return -2;
    uint64_t blocks=(uint64_t(seqend)+15)/16, padded=(uint64_t(seqend)+63)/64*64;
    uint64_t ni=uint64_t(pblocks)*16*512*2, nb=blocks*4, no=padded*656;
    if(btcount<blocks||outbytes<no) return -3;
    if(!aligned(paged,2)||!aligned(bt,4)||!aligned(out,4)||
       overlap(paged,ni,out,no)||overlap(bt,nb,out,no)) return -4;
    prep_kv<<<unsigned(padded),128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(paged),bt,static_cast<unsigned char*>(out),seqend);
    return int(cudaGetLastError());
}
