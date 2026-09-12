// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include "adapter.cuh"

// External library entry points (linked later, no dlopen).
extern "C" long long atlas_flash_kda_workspace_size(int T, int H, int N);
extern "C" int atlas_flash_kda_prefill_fp32_state(
    const void* q, const void* k, const void* v, const void* gate,
    const void* beta, void* state, void* out, void* workspace,
    const void* a, const void* bias, const int64_t* cu, const int* slots,
    int T, int H, int N, int capacity, float scale, float lower,
    void* stream);

// Writes cu[0]=row_start=0, cu[1]=row_end=T, slots[0]=sequence slot=0.
__global__ void atlas_mango_meta_kernel(int64_t* cu, int* slots, int T) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cu[0] = 0;
        cu[1] = (int64_t)T;
        slots[0] = 0;
    }
}

// Canonical state layout: [key(128)][value(128)] per (head,row) region.
// adapter.cuh provides atlas_state_transpose(src,dst,heads) with grid(4,4,H) block(32,8).

static inline int ptrs_ok(const void* qkv, const void* gate, const void* rawbeta,
                          const void* a, const void* bias, void* state, void* out,
                          void* packed, void* workspace, void* aux) {
    return qkv && gate && rawbeta && a && bias && state && out &&
           packed && workspace && aux;
}

extern "C" int atlas_mango_flash_prefill(
    const void* qkv, const void* gate, const void* rawbeta,
    const void* a, const void* bias, void* state, void* out,
    void* packed, void* workspace, void* aux,
    uint64_t packed_bytes, uint64_t workspace_bytes, uint64_t aux_bytes,
    int T, int H, float scale, float lower, void* stream) {

    if (!ptrs_ok(qkv, gate, rawbeta, a, bias, state, out, packed, workspace, aux))
        return -1;
    if (T < 2048 || T > 4100) return -1;
    if (H != 32) return -1;
    if (!(scale > 0.0f) || !std::isfinite(scale)) return -1;
    if (lower != -5.0f || lower != lower) return -1;     // exactly -5.0

    const uint64_t plane = (uint64_t)T * (uint64_t)H * 128u * 2u;   // bytes/plane
    if (packed_bytes < 3u * plane) return -1;

    const uint64_t state_bytes = (uint64_t)H * 128u * 128u * 4u;    // float state
    const uint64_t beta_bytes = (uint64_t)H * (uint64_t)T * 2u;     // raw beta
    uint64_t off = state_bytes + beta_bytes;
    off = (off + 127u) & ~(uint64_t)127u;                            // align up to 128
    const uint64_t cu_off = off;
    const uint64_t slots_off = cu_off + 16u;                         // int64 cu[2]
    const uint64_t req_aux = slots_off + 4u;                         // int32 slots[1]
    if (aux_bytes < req_aux) return -1;

    const long long required_workspace = atlas_flash_kda_workspace_size(T, H, 1);
    if (required_workspace <= 0 || workspace_bytes < (uint64_t)required_workspace) return -1;

    cudaStream_t s = stream ? (cudaStream_t)stream : (cudaStream_t)0;

    int64_t* cu = (int64_t*)((char*)aux + cu_off);
    int* slots = (int*)((char*)aux + slots_off);

    atlas_mango_meta_kernel<<<1, 1, 0, s>>>(cu, slots, T);
    int status = (int)cudaGetLastError();
    if (status) return status;

    void* beta_ht = (char*)aux + state_bytes;
    void* p = packed;
    void* pq = p;
    void* pk = (char*)p + plane;
    void* pv = (char*)p + 2u * plane;
    atlas_flash_pack<<<dim3(H, T), 128, 0, s>>>(
        (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)rawbeta,
        (__nv_bfloat16*)pq, (__nv_bfloat16*)pk, (__nv_bfloat16*)pv,
        (__nv_bfloat16*)beta_ht, (unsigned)T, (unsigned)H);
    status = (int)cudaGetLastError();
    if (status) return status;

    atlas_state_transpose<<<dim3(4, 4, H), dim3(32, 8), 0, s>>>(
        (float*)state, (float*)aux, (unsigned)H);
    status = (int)cudaGetLastError();
    if (status) return status;

    int rc = atlas_flash_kda_prefill_fp32_state(
        pq, pk, pv, gate, beta_ht, aux, out, workspace,
        a, bias, cu, slots, T, H, 1, 1, scale, lower, (void*)s);
    if (rc != 0) return rc;
    status = (int)cudaGetLastError();
    if (status) return status;

    atlas_state_transpose<<<dim3(4, 4, H), dim3(32, 8), 0, s>>>(
        (float*)aux, (float*)state, (unsigned)H);
    status = (int)cudaGetLastError();
    if (status) return status;

    return 0;
}

extern "C" int atlas_mango_flash_abi_version(void) { return 1; }
