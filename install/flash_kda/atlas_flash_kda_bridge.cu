// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_runtime.h>

#include <cstdint>

#include "fwd.h"

extern "C" long long atlas_flash_kda_workspace_size(int total_tokens,
                                                     int heads,
                                                     int sequences) {
  if (total_tokens <= 0 || heads <= 0 || sequences <= 0) return -1;
  constexpr long long chunk = 16;
  constexpr long long dimension = 128;
  const long long total_tiles =
      (total_tokens + chunk - 1) / chunk + sequences;
  const long long per_tile = 3 * chunk * dimension * 2 + dimension * 4 +
                             2 * chunk * chunk * 2;
  const long long prefix = ((sequences + 1) * 4 + 127) / 128 * 128;
  return heads * total_tiles * per_tile + prefix;
}

// Atlas-native, inference-only FlashKDA entry. Inputs are already laid out as
// packed varlen [total_tokens,heads,128]. beta_ht is the one exceptional
// [heads,total_tokens] input required by FlashKDA's 1D TMA descriptor.
// Recurrent state is one FP32 [state_capacity,heads,value=128,key=128] pool and
// is updated in place through logical-sequence -> physical-slot IDs, matching
// Atlas decode's persistent ABI without a multi-megabyte gather/scatter.
extern "C" int atlas_flash_kda_prefill_fp32_state(
    const void* query,
    const void* key,
    const void* value,
    const void* forget,
    const void* beta_ht,
    void* recurrent_state,
    void* output,
    void* workspace,
    const void* a_log,
    const void* dt_bias,
    const std::int64_t* cu_seqlens,
    const int* state_slot_ids,
    int total_tokens,
    int heads,
    int sequences,
    int state_capacity,
    float query_scale,
    float lower_bound,
    void* stream_raw) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      forget == nullptr || beta_ht == nullptr || recurrent_state == nullptr ||
      output == nullptr || workspace == nullptr || a_log == nullptr ||
      dt_bias == nullptr || cu_seqlens == nullptr || state_slot_ids == nullptr ||
      total_tokens <= 0 || heads <= 0 || sequences <= 0 ||
      state_capacity < sequences || lower_bound > 0.0f) {
    return -1;
  }
  constexpr int chunk = 16;
  const int total_tiles =
      (total_tokens + chunk - 1) / chunk + sequences;
  const float gate_scale = lower_bound * 1.4426950408889634f;
  launch_fwd<128, true, true, true, true>(
      static_cast<const cutlass::bfloat16_t*>(query),
      static_cast<const cutlass::bfloat16_t*>(key),
      static_cast<const cutlass::bfloat16_t*>(value),
      static_cast<const cutlass::bfloat16_t*>(forget),
      static_cast<const cutlass::bfloat16_t*>(beta_ht), recurrent_state,
      query_scale, recurrent_state,
      static_cast<cutlass::bfloat16_t*>(output), workspace, total_tiles,
      total_tokens, heads, sequences, cu_seqlens,
      static_cast<const float*>(a_log), static_cast<const float*>(dt_bias),
      gate_scale, state_slot_ids, state_capacity,
      reinterpret_cast<cudaStream_t>(stream_raw));
  return static_cast<int>(cudaPeekAtLastError());
}
