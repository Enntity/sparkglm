// SPDX-License-Identifier: AGPL-3.0-only

// Small ctypes bridge for the raw Atlas EXL3 kernel. This is a correctness
// probe only; production uses the CUDA-driver launch path in spark-runtime.

#include <cuda_runtime.h>
#include <stdint.h>

#include "../../kernels/gb10/glm-5.3-flash/exl3/glm53_exl3_moe.cu"

extern "C" int atlas_glm53_exl3_launch(
    uintptr_t hidden_state,
    uintptr_t temp_state_g,
    uintptr_t temp_state_u,
    uintptr_t temp_intermediate_g,
    uintptr_t temp_intermediate_u,
    uintptr_t output_state,
    uintptr_t gate_trellis,
    uintptr_t gate_suh,
    uintptr_t gate_svh,
    uintptr_t up_trellis,
    uintptr_t up_suh,
    uintptr_t up_svh,
    uintptr_t down_trellis,
    uintptr_t down_suh,
    uintptr_t down_svh,
    uintptr_t expert_count,
    uintptr_t token_sorted,
    uintptr_t weight_sorted,
    int hidden_dim,
    int intermediate_dim,
    int num_experts,
    int num_experts_per_tok,
    int max_tokens_per_expert,
    int concurrency,
    float act_limit,
    uintptr_t locks
) {
    cudaError_t status = cudaFuncSetAttribute(
        glm53_exl3_moe,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_MAX
    );
    if (status != cudaSuccess) return static_cast<int>(status);

    glm53_exl3_moe<<<dim3(MOE_SMS_PER_EXPERT, 1, concurrency), 512, SMEM_MAX>>>(
        reinterpret_cast<const half*>(hidden_state),
        reinterpret_cast<half*>(temp_state_g),
        reinterpret_cast<half*>(temp_state_u),
        reinterpret_cast<half*>(temp_intermediate_g),
        reinterpret_cast<half*>(temp_intermediate_u),
        reinterpret_cast<float*>(output_state),
        reinterpret_cast<const uint16_t**>(gate_trellis),
        reinterpret_cast<const half**>(gate_suh),
        reinterpret_cast<const half**>(gate_svh),
        reinterpret_cast<const uint16_t**>(up_trellis),
        reinterpret_cast<const half**>(up_suh),
        reinterpret_cast<const half**>(up_svh),
        reinterpret_cast<const uint16_t**>(down_trellis),
        reinterpret_cast<const half**>(down_suh),
        reinterpret_cast<const half**>(down_svh),
        reinterpret_cast<const int64_t*>(expert_count),
        reinterpret_cast<const int64_t*>(token_sorted),
        reinterpret_cast<const half*>(weight_sorted),
        hidden_dim,
        intermediate_dim,
        num_experts,
        num_experts_per_tok,
        max_tokens_per_expert,
        concurrency,
        act_limit,
        MOE_ACT_SILU,
        4,
        4,
        4,
        reinterpret_cast<int*>(locks)
    );
    return static_cast<int>(cudaGetLastError());
}
