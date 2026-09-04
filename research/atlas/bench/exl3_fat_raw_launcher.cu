// SPDX-License-Identifier: AGPL-3.0-only

// Ctypes bridge for the device-compacted wide-row EXL3 path.

#include <cuda_runtime.h>
#include <stdint.h>

#include "../../kernels/gb10/glm-5.3-flash/exl3/glm53_exl3_fat.cu"

extern "C" int atlas_glm53_exl3_fat_launch(
    uintptr_t hidden,
    uintptr_t transformed,
    uintptr_t gate_up,
    uintptr_t output,
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
    uintptr_t descriptors,
    uintptr_t token_sorted,
    uintptr_t weight_sorted,
    int hidden_size,
    int intermediate_size,
    int num_experts,
    int rows,
    float activation_limit) {
  dim3 persistent_grid(48);
  constexpr int shared_bytes = 13 * 1024;
  auto* transformed_ptr = reinterpret_cast<half*>(transformed);
  auto* counts_ptr = reinterpret_cast<const int64_t*>(expert_count);
  auto* descriptors_ptr = reinterpret_cast<const int4*>(descriptors);
  auto gather = [&](uintptr_t suh) {
    glm53_exl3_fat_gather<<<persistent_grid, 256>>>(
        reinterpret_cast<const half*>(hidden), transformed_ptr,
        reinterpret_cast<const half**>(suh), counts_ptr, descriptors_ptr,
        reinterpret_cast<const int64_t*>(token_sorted), hidden_size, rows,
        num_experts);
  };
  auto gemm = [&](uintptr_t trellis, uintptr_t svh, int column) {
    glm53_exl3_fat_gemm_gate_up<<<persistent_grid, 256, shared_bytes>>>(
        transformed_ptr, reinterpret_cast<const uint16_t**>(trellis),
        reinterpret_cast<float*>(gate_up),
        reinterpret_cast<const half**>(svh), counts_ptr, descriptors_ptr,
        hidden_size, intermediate_size, 2 * intermediate_size, column, rows,
        num_experts);
  };
  gather(gate_suh);
  gemm(gate_trellis, gate_svh, 0);
  gather(up_suh);
  gemm(up_trellis, up_svh, intermediate_size);
  glm53_exl3_fat_activate_down_had<<<persistent_grid, 256>>>(
      reinterpret_cast<const float*>(gate_up), transformed_ptr,
      reinterpret_cast<const half**>(down_suh), counts_ptr, descriptors_ptr,
      intermediate_size, 2 * intermediate_size, activation_limit, rows,
      num_experts);
  glm53_exl3_fat_gemm_down_scatter<<<persistent_grid, 256, shared_bytes>>>(
      transformed_ptr, reinterpret_cast<const uint16_t**>(down_trellis),
      reinterpret_cast<float*>(output),
      reinterpret_cast<const half**>(down_svh), counts_ptr, descriptors_ptr,
      reinterpret_cast<const int64_t*>(token_sorted),
      reinterpret_cast<const half*>(weight_sorted), intermediate_size,
      hidden_size, rows, num_experts);
  return static_cast<int>(cudaGetLastError());
}
