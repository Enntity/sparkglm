// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// No init/shutdown: caller owns device/context, stream, allocations and lifetime.
int atlas_sparse_native_version(void);
// Dense BF16 q[rows,heads,576], packed uint8 kv[pages,64,656],
// int32 indices[rows,2048], BF16 out[rows,heads,512], FP32 lse[rows,heads],
// int32 lengths[rows] in [0,2048]. All pointers required except stream (0=default).
// Caller guarantees device residency, capacities, valid indices and disjoint
// output/input ranges. Async: buffers survive stream completion; no sync here.
// Returns 0 on launch, positive cudaError_t, or negative bridge validation code:
// -1=null, -2=rows, -3=heads, -4=scale, -5=unsupported dispatch, -6=C++ exception.
// NOTE: prebuilt upstream CUDA_CHECK may abort the process before returning.
int atlas_sparse_native_launch_glm(const void* q576, const void* kv656,
    const int32_t* indices2048, void* out512, float* lse, int32_t rows,
    int32_t heads, float scale, const int32_t* lengths, void* stream);
#ifdef __cplusplus
}
#endif
