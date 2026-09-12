// SPDX-License-Identifier: AGPL-3.0-only
// Mechanical C ABI wrapper. Kernel is provided by the included header; do not modify.
#include "kernels/gb10/deepseek-v4-flash/nvfp4/glm_sparse_prefill_kv_reuse.cu"

extern "C" int atlas_sparse_init(void) {
  return (int)cudaFuncSetAttribute(
      (const void*)glm_sparse_mla_prefill_bf16_head32_tc_kv_pad,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 69376);
}

extern "C" int atlas_sparse_run(const void* q, const void* kv, const void* indices,
                                void* out, const void* table, unsigned rows,
                                unsigned heads, unsigned width, unsigned block_size,
                                float scale, void* stream) {
  if (q == nullptr || kv == nullptr || indices == nullptr || out == nullptr ||
      table == nullptr) {
    return (int)cudaErrorInvalidValue;
  }
  if (rows == 0 || heads != 32 || width == 0 || block_size == 0) {
    return (int)cudaErrorInvalidValue;
  }
  glm_sparse_mla_prefill_bf16_head32_tc_kv_pad<<<dim3(1, rows), 256, 69376,
      (cudaStream_t)stream>>>(
      (const __nv_bfloat16*)q, (const __nv_bfloat16*)kv, (const __nv_bfloat16*)kv,
      (const int*)indices, (__nv_bfloat16*)out, (const unsigned*)table, rows, heads, 512u, width, block_size,
      scale);
  return (int)cudaGetLastError();
}
