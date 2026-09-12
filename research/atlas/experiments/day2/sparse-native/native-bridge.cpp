// SPDX-License-Identifier: AGPL-3.0-only
#include "native-bridge.h"
#include "reference/traits/model/model_type.h" // Exact NVIDIA BSD-3-Clause header.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstddef>
// Declaration matches reference/sparse_mla_sm120_prefill.cu:397-404 verbatim
// in types and parameter order; bf16 there is the global __nv_bfloat16 alias.
namespace flashinfer::sparse_mla_sm120 {
bool sparse_mla_prefill_dispatch(ModelType mt, int num_heads, int topk, int page_block_size,
    int topk_extra, int extra_page_block_size, const __nv_bfloat16* Q,
    const uint8_t* KV_cache, const int32_t* indices, const uint8_t* extra_KV_cache,
    const int32_t* extra_indices, __nv_bfloat16* output, float* out_lse, float sm_scale,
    int num_tokens, size_t stride_kv_block, size_t stride_kv_block_extra,
    const float* attn_sink, const int* topk_length, const int* extra_topk_length,
    cudaStream_t stream);
}
static_assert(static_cast<int>(ModelType::GLM_NSA)==2);
static_assert(sizeof(int)==sizeof(int32_t));
extern "C" int atlas_sparse_native_version(void) { return 1; }
extern "C" int atlas_sparse_native_launch_glm(const void* q, const void* kv,
    const int32_t* indices, void* out, float* lse, int32_t rows, int32_t heads,
    float scale, const int32_t* lengths, void* stream) {
    if(!q || !kv || !indices || !out || !lse || !lengths) return -1;
    if(rows<=64 || rows>INT_MAX/4) return -2; // Max four head groups per row.
    if(heads!=8 && heads!=16 && heads!=32 && heads!=64 && heads!=128) return -3;
    if(!std::isfinite(scale) || scale<=0) return -4;
    try {
        bool dispatched=flashinfer::sparse_mla_sm120::sparse_mla_prefill_dispatch(
            ModelType::GLM_NSA,heads,2048,64,0,0,
            static_cast<const __nv_bfloat16*>(q),static_cast<const uint8_t*>(kv),indices,
            nullptr,nullptr,static_cast<__nv_bfloat16*>(out),lse,scale,rows,
            size_t(64)*656,0,nullptr,lengths,nullptr,static_cast<cudaStream_t>(stream));
        if(!dispatched) return -5;
        return static_cast<int>(cudaGetLastError());
    } catch(...) { return -6; }
}
