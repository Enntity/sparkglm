// SPDX-License-Identifier: AGPL-3.0-only

// ctypes bridge for the real-checkpoint FP32 router parity probe.

#include <cuda_runtime.h>
#include <stdint.h>

#include "dense_gemm_bf16.cu"
#include "glm53_router.cu"

extern "C" int atlas_glm53_router_launch(
    uintptr_t hidden,
    uintptr_t gate_weight,
    uintptr_t correction_bias,
    uintptr_t logits,
    uintptr_t expert_indices,
    uintptr_t expert_weights,
    int rows,
    uintptr_t stream_raw) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  dense_gemm_bf16_f32out<<<dim3(18, (rows + 15) / 16), dim3(16, 16), 0,
                              stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden),
      reinterpret_cast<const __nv_bfloat16*>(gate_weight),
      reinterpret_cast<float*>(logits), rows, 288, 4096);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return static_cast<int>(status);
  glm53_moe_topk_sigmoid_batched_f32<<<rows, 256, 0, stream>>>(
      reinterpret_cast<const float*>(logits),
      reinterpret_cast<const float*>(correction_bias),
      reinterpret_cast<unsigned int*>(expert_indices),
      reinterpret_cast<float*>(expert_weights), 288, 8, 1, 2.5f);
  return static_cast<int>(cudaGetLastError());
}
