// SPDX-License-Identifier: AGPL-3.0-only
// CPU-only shim for the exact retained GLM32 CUDA host stub; no kernel build.
#include "reference/traits/model/model_type.h" // NVIDIA BSD-3-Clause, unchanged.
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

// These types/templates are GLOBAL in the NVIDIA headers, not inside the
// dispatcher's flashinfer::sparse_mla_sm120 namespace. A complete cold-parameter
// definition is unnecessary: this shim takes an address and never calls it.
struct PrefillColdParams;
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK,
          int PAGE_BLOCK_SIZE, int MG_N_HG_T>
void sparse_mla_prefill_mg_kernel(const __nv_bfloat16*, const uint8_t*,
                                const int32_t*, __nv_bfloat16*, float*,
                                const float*, PrefillColdParams);

namespace {
static_assert(static_cast<int>(ModelType::GLM_NSA) == 2);
static_assert(static_cast<int>(ComputeMode::FP8) == 0);
static_assert(sizeof(uint8_t) == 1 && sizeof(int32_t) == 4);

// Exact SmemLayoutMG<GLM_NSA, FP8>::TOTAL from retained traits:
// 2 Q-nope buffers + 2 Q-scale buffers + 2 KV buffers + shared reduction
// + M/L + W scales + two-parity W FP8 + two mbarriers (all sizes in bytes).
// GLM has no separate KV-scale staging; inline scales are copied with KV.
constexpr int kSelectedDynamicSmem =
    2 * (16 * 528) + 2 * (16 * 4 * 4) + 2 * (64 * 528) +
    (2 * 8 * 16 * 4) + 2 * (2 * 16 * 4) + (2 * 4 * 16 * 4) +
    (2 * 2 * 16 * (64 + 16)) + 2 * 8;
static_assert(kSelectedDynamicSmem == 91920);
}

// Call on each rank's current device/context BEFORE arena/KV sizing.
// No cudaMalloc, work launch, device/context change, or synchronization.
// Loading code can itself consume device memory; this is the purpose of init.
// Do not cache globally: success on one context does not initialize another.
extern "C" __attribute__((visibility("default")))
int atlas_sparse_native_init_selected(void) {
  const void* kernel = reinterpret_cast<const void*>(
      &sparse_mla_prefill_mg_kernel<ModelType::GLM_NSA, ComputeMode::FP8,
                                  32, 2048, 64, 2>);
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) return static_cast<int>(status);
  status = cudaFuncSetAttribute(kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSelectedDynamicSmem);
  return static_cast<int>(status);
}
