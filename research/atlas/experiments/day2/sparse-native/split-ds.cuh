// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

__global__ void split_ids(const int32_t* __restrict__ src,
                          int32_t* __restrict__ main_ids,
                          int32_t* __restrict__ tail_ids,
                          int32_t* __restrict__ main_lengths,
                          int32_t* __restrict__ native_tail_lengths,
                          int32_t* __restrict__ true_tail_counts,
                          unsigned rows, unsigned seq_start) {
    const unsigned r = blockIdx.x;
    if (r >= rows) return;
    const unsigned tid = threadIdx.x;

    const size_t src_row = (size_t)r * 2051u;
    const size_t dst_row = (size_t)r * 2048u;

    for (unsigned i = tid; i < 2048u; i += 256u)
        main_ids[dst_row + i] = src[src_row + i];

    const int32_t true_count = (int32_t)(((size_t)seq_start + r + 1u) & 3u);
    const int32_t native_tail = true_count > 1 ? true_count : 1;

    for (unsigned i = tid; i < 2048u; i += 256u)
        tail_ids[dst_row + i] = -1;

    if (true_count == 0) {
        if (tid == 0) tail_ids[dst_row] = 0;
    } else if (tid < (unsigned)true_count) {
        tail_ids[dst_row + tid] = src[src_row + 2048u + tid];
    }

    if (tid == 0) {
        main_lengths[r] = 2048;
        native_tail_lengths[r] = native_tail;
        true_tail_counts[r] = true_count;
    }
}
