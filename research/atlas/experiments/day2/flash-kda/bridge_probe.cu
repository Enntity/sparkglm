// SPDX-License-Identifier: AGPL-3.0-only
// Standalone glue over Mango Atlas dd0ffd157fb7c6e96d2f3858c888513c6f142cf6.
// Compile with -I<atlas>/kernels/gb10/common; do not substitute a baseline.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "kda.cu"
#include "adapter.cuh"

using BF16 = __nv_bfloat16;

static bool valid_shape(unsigned int tokens, unsigned int heads) {
    return tokens > 0 && tokens <= 65535 && heads > 0 && heads <= 65535;
}

// GLM D=128 and lower_bound=-5.0; state is FP32 [H,key,value].
// Scratch qn/kn/decay: T*H*128 floats each; beta: T*H floats.
extern "C" int atlas_probe_baseline(
    const void* qkv, const void* raw_gate, const void* raw_beta,
    const void* a_log, const void* dt_bias,
    void* qn, void* kn, void* decay, void* beta, void* state, void* out,
    unsigned int tokens, unsigned int heads, void* stream_raw) {
    if (!valid_shape(tokens, heads) || !qkv || !raw_gate || !raw_beta ||
        !a_log || !dt_bias || !qn || !kn || !decay || !beta || !state || !out)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    kda_preprocess_regresident<<<dim3(heads, tokens), 128, 0, stream>>>(
        static_cast<const BF16*>(qkv), static_cast<const BF16*>(raw_gate),
        static_cast<const BF16*>(raw_beta), static_cast<const float*>(a_log),
        static_cast<const float*>(dt_bias), static_cast<float*>(qn),
        static_cast<float*>(kn), static_cast<float*>(decay),
        static_cast<float*>(beta), tokens, heads, 128, -5.0f);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    kda_recurrent_bf16_regresident<<<dim3(heads, 32), 128, 0, stream>>>(
        static_cast<const BF16*>(qkv), static_cast<const float*>(qn),
        static_cast<const float*>(kn), static_cast<const float*>(decay),
        static_cast<const float*>(beta), static_cast<float*>(state),
        static_cast<BF16*>(out), tokens, heads, 128);
    return static_cast<int>(cudaGetLastError());
}

// Q/K are raw BF16; FlashKDA normalizes internally and applies 1/sqrt(128).
// q/k/v each hold T*H*128 BF16, beta_ht holds H*T raw BF16 logits.
extern "C" int atlas_probe_prepare(
    const void* qkv, const void* raw_beta,
    void* q, void* k, void* v, void* beta_ht,
    unsigned int tokens, unsigned int heads, void* stream_raw) {
    if (!valid_shape(tokens, heads) || !qkv || !raw_beta ||
        !q || !k || !v || !beta_ht)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    atlas_flash_pack<<<dim3(heads, tokens), 128, 0, stream>>>(
        static_cast<const BF16*>(qkv), static_cast<const BF16*>(raw_beta),
        static_cast<BF16*>(q), static_cast<BF16*>(k), static_cast<BF16*>(v),
        static_cast<BF16*>(beta_ht), tokens, heads);
    return static_cast<int>(cudaGetLastError());
}

// Out-of-place H*128*128 FP32 transpose in either direction.
// Adapter contract: tile[32][33], block(32,8), j=0,8,16,24 only.
extern "C" int atlas_probe_transpose(
    const void* src, void* dst, unsigned int heads, void* stream_raw) {
    if (!valid_shape(1, heads) || !src || !dst || src == dst)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    atlas_state_transpose<<<dim3(4, 4, heads), dim3(32, 8), 0, stream>>>(
        static_cast<const float*>(src), static_cast<float*>(dst), heads);
    return static_cast<int>(cudaGetLastError());
}
