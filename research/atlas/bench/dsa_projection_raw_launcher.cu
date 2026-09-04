// SPDX-License-Identifier: AGPL-3.0-only

// ctypes bridge for the checkpoint-sized BF16 DSA projection parity probe.

#include <cuda_runtime.h>
#include <stdint.h>

#include "glm53_dsa_projection.cu"

extern "C" int atlas_glm53_dsa_projection_launch(
    uintptr_t query,
    uintptr_t latent,
    uintptr_t kv_b_weight,
    uintptr_t absorbed,
    uintptr_t expanded,
    int num_tokens,
    uintptr_t stream_raw) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
  glm53_dsa_absorb_query_bf16<<<dim3(64, num_tokens), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),
      reinterpret_cast<const __nv_bfloat16*>(kv_b_weight),
      reinterpret_cast<__nv_bfloat16*>(absorbed), num_tokens);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return static_cast<int>(status);
  glm53_dsa_expand_value_bf16<<<dim3(64, num_tokens), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(latent),
      reinterpret_cast<const __nv_bfloat16*>(kv_b_weight),
      reinterpret_cast<__nv_bfloat16*>(expanded), num_tokens);
  return static_cast<int>(cudaGetLastError());
}
