// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <climits>

// Gathers (scatter=0): segments -> packed. Scatter(1): packed -> segments.
__global__ void gather_scatter_kernel(const uint4* __restrict__ p0,
                                      const uint4* __restrict__ p1,
                                      const uint4* __restrict__ p2,
                                      const uint4* __restrict__ p3,
                                      uint4* __restrict__ packed,
                                      size_t nvec, size_t segvec,
                                      int scatter) {
    const uint4* const srcs[4] = {p0, p1, p2, p3};
    uint4* const dsts[4] = {
        const_cast<uint4*>(p0), const_cast<uint4*>(p1),
        const_cast<uint4*>(p2), const_cast<uint4*>(p3)};
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < nvec; i += stride) {
        const size_t s = i / segvec;          // segment index 0..3
        const size_t local = i - s * segvec;  // vector offset in segment
        if (scatter) {
            packed[i] = dsts[s][local];
        } else {
            dsts[s][local] = packed[i];
        }
    }
}

static bool ranges_disjoint(const void* a[5], size_t n) {
    const uintptr_t base[5] = {(uintptr_t)a[0], (uintptr_t)a[1],
                               (uintptr_t)a[2], (uintptr_t)a[3],
                               (uintptr_t)a[4]};
    for (int i = 0; i < 5; ++i) {
        if (base[i] == 0 || (base[i] & 0xF) != 0) return false;
        if (base[i] > UINTPTR_MAX - n) return false;
    }
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j) {
            const uintptr_t ai = base[i], aj = base[j];
            if (ai < aj) {
                if (ai + n > aj) return false;
            } else {
                if (aj + n > ai) return false;
            }
        }
    return true;
}

extern "C" int copy_four_segments(void* p0, void* p1, void* p2, void* p3,
                                  void* packed, size_t segment_bytes,
                                  int scatter, void* stream) {
    const void* addrs[5] = {p0, p1, p2, p3, packed};
    if ((scatter != 0 && scatter != 1) ||
        segment_bytes == 0 || (segment_bytes & 0xF) != 0 ||
        segment_bytes > (size_t)(16 * 1024 * 1024))
        return cudaErrorInvalidValue;
    if (packed == nullptr) return cudaErrorInvalidValue;
    if (segment_bytes > SIZE_MAX / 4) return cudaErrorInvalidValue;
    const size_t total = 4 * segment_bytes;
    if (!ranges_disjoint(addrs, total)) return cudaErrorInvalidValue;

    const size_t nvec = total / 16;
    const size_t segvec = segment_bytes / 16;
    int grid = (int)((nvec + 255) / 256);
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    gather_scatter_kernel<<<grid, 256, 0, s>>>(
        (const uint4*)p0, (const uint4*)p1, (const uint4*)p2,
        (const uint4*)p3, (uint4*)packed, nvec, segvec, scatter);
    return (int)cudaGetLastError();
}
