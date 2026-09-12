// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// ABI1, sizeof112/alignment8. Caller owns CUDA device/context/stream and spans.
// All ten device spans must be disjoint and survive stream completion. Rust
// checks actual allocation capacities; C checks geometry, alignment and spans.
typedef struct AtlasGlmSparseNativeArgs {
    uint64_t q, kv, selected, table, qpad, packed_kv, metadata;
    uint64_t main_out, tail_out, out;
    uint64_t metadata_bytes, stream;
    uint32_t rows, seq_start, physical_blocks, block_table_count;
} AtlasGlmSparseNativeArgs;
int atlas_glm_sparse_native_version(void);
// Call on the serving CUDA context before KV free-memory sizing.
int atlas_glm_sparse_native_init(void);
// Asynchronous, no allocations/synchronization. Error after work submission is
// fatal to this forward pass: caller must never retry with another implementation.
// Selected IDs and block table are canonical, valid device data supplied by Atlas.
// Return0 success, positive CUDA code, or -1 null/-2 geometry/-3 metadata/-4 spans.
int atlas_glm_sparse_native_run(const AtlasGlmSparseNativeArgs* args);
#ifdef __cplusplus
}
static_assert(sizeof(AtlasGlmSparseNativeArgs)==112);
static_assert(alignof(AtlasGlmSparseNativeArgs)==8);
#endif
