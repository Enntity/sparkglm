// SPDX-License-Identifier: AGPL-3.0-only
#include "atlas-glm-sparse-native.h"
#include "native-bridge.h"
#include "native-prep.cu"
#include "split-ds.cuh"
#include "merge-ds.cu"
extern "C" int atlas_sparse_native_init_selected(void);
namespace {
uint64_t align256(uint64_t n) { return (n+255)&~uint64_t(255); }
void* ptr(uint64_t n) { return reinterpret_cast<void*>(n); }
struct Metadata {
    int32_t *main_ids, *tail_ids, *main_lengths, *tail_lengths, *tail_counts;
    float *main_lse, *tail_lse, *final_lse;
    uint64_t bytes;
    Metadata(uint64_t base, uint32_t rows) {
        uint64_t offset=0;
        auto region=[&](uint64_t bytes) {
            offset=align256(offset); uint64_t address=base+offset;
            offset+=bytes; return address;
        };
        main_ids=reinterpret_cast<int32_t*>(region(uint64_t(rows)*2048*4));
        tail_ids=reinterpret_cast<int32_t*>(region(uint64_t(rows)*2048*4));
        main_lengths=reinterpret_cast<int32_t*>(region(uint64_t(rows)*4));
        tail_lengths=reinterpret_cast<int32_t*>(region(uint64_t(rows)*4));
        tail_counts=reinterpret_cast<int32_t*>(region(uint64_t(rows)*4));
        main_lse=reinterpret_cast<float*>(region(uint64_t(rows)*32*4));
        tail_lse=reinterpret_cast<float*>(region(uint64_t(rows)*32*4));
        final_lse=reinterpret_cast<float*>(region(uint64_t(rows)*32*4));
        bytes=offset;
    }
};
}
extern "C" int atlas_glm_sparse_native_version(void) { return 1; }
extern "C" int atlas_glm_sparse_native_init(void) {
    int status=atlas_sparse_native_init_selected();
    if(status) return status;
    cudaFuncAttributes attr{};
    const void* functions[]={reinterpret_cast<const void*>(prep_q),
        reinterpret_cast<const void*>(prep_kv),reinterpret_cast<const void*>(split_ids),
        reinterpret_cast<const void*>(atlas_sparse_merge_kernel)};
    for(const void* function:functions) {
        auto result=cudaFuncGetAttributes(&attr,function);
        if(result!=cudaSuccess) return int(result);
    }
    return 0;
}
extern "C" int atlas_glm_sparse_native_run(const AtlasGlmSparseNativeArgs* a) {
    if(!a) return -1;
    if(a->rows<2048 || a->rows>4100 || a->seq_start<2048 ||
       a->seq_start>32768-a->rows || !a->physical_blocks || a->physical_blocks>32768)
        return -2;
    uint32_t end=a->seq_start+a->rows;
    if(a->block_table_count<(end+15)/16) return -2;
    // Geometry is bounded before arithmetic. Check every live input/output span
    // before any CUDA call; each metadata offset below is relative to one owner.
    Metadata layout(0,a->rows);
    if(a->metadata_bytes<layout.bytes) return -3;
    uint64_t qbytes=uint64_t(a->rows)*32*512*2;
    uint64_t qpadbytes=uint64_t(a->rows)*32*576*2;
    uint64_t kvbytes=uint64_t((end+63)/64)*64*656;
    const uint64_t addresses[]={a->q,a->kv,a->selected,a->table,a->qpad,
        a->packed_kv,a->metadata,a->main_out,a->tail_out,a->out};
    const uint64_t sizes[]={qbytes,uint64_t(a->physical_blocks)*16*512*2,
        uint64_t(a->rows)*2051*4,uint64_t((end+15)/16)*4,qpadbytes,
        kvbytes,layout.bytes,qbytes,qbytes,qbytes};
    const uint64_t alignment[]={2,2,4,4,256,256,256,256,256,256};
    for(unsigned i=0;i<10;++i) {
        if(!addresses[i]) return -1;
        if(addresses[i]%alignment[i] || sizes[i]>UINT64_MAX-addresses[i]) return -4;
        for(unsigned j=0;j<i;++j)
            if(addresses[i]<addresses[j]+sizes[j] && addresses[j]<addresses[i]+sizes[i]) return -4;
    }
    Metadata m(a->metadata,a->rows);
    auto stream=reinterpret_cast<cudaStream_t>(a->stream);
    int result=atlas_native_prep_q(ptr(a->q),ptr(a->qpad),a->rows,qbytes,qpadbytes,stream);
    if(result) return result;
    result=atlas_native_prep_kv(ptr(a->kv),a->physical_blocks,
        reinterpret_cast<const int32_t*>(a->table),a->block_table_count,
        ptr(a->packed_kv),kvbytes,end,stream);
    if(result) return result;
    split_ids<<<a->rows,256,0,stream>>>(reinterpret_cast<const int32_t*>(a->selected),
        m.main_ids,m.tail_ids,m.main_lengths,m.tail_lengths,m.tail_counts,a->rows,a->seq_start);
    result=int(cudaGetLastError()); if(result) return result;
    result=atlas_sparse_native_launch_glm(ptr(a->qpad),ptr(a->packed_kv),m.main_ids,
        ptr(a->main_out),m.main_lse,a->rows,32,0.0625f,m.main_lengths,stream);
    if(result) return result;
    result=atlas_sparse_native_launch_glm(ptr(a->qpad),ptr(a->packed_kv),m.tail_ids,
        ptr(a->tail_out),m.tail_lse,a->rows,32,0.0625f,m.tail_lengths,stream);
    if(result) return result;
    return atlas_sparse_merge(ptr(a->main_out),ptr(a->tail_out),m.main_lse,m.tail_lse,
        m.tail_counts,ptr(a->out),m.final_lse,a->rows,32,stream);
}
