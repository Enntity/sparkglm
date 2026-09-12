// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <climits>

namespace {

__global__ void four_segment_kernel(unsigned char* __restrict__ p0,
                                    unsigned char* __restrict__ p1,
                                    unsigned char* __restrict__ p2,
                                    unsigned char* __restrict__ p3,
                                    unsigned char* __restrict__ packed,
                                    size_t seg_vecs, int scatter, size_t total_vecs) {
    const size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
    for (size_t v = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
                   static_cast<size_t>(threadIdx.x);
         v < total_vecs; v += stride) {
        const size_t seg = v / seg_vecs;
        const size_t off = (v - seg * seg_vecs) * sizeof(uint4);
        unsigned char* dst;
        const unsigned char* src;
        if (scatter) {
            dst = (seg == 0) ? p0 : (seg == 1) ? p1 : (seg == 2) ? p2 : p3;
            dst += off;
            src = packed + v * sizeof(uint4);
        } else {
            dst = packed + v * sizeof(uint4);
            const unsigned char* base =
                (seg == 0) ? p0 : (seg == 1) ? p1 : (seg == 2) ? p2 : p3;
            src = base + off;
            dst = packed + v * sizeof(uint4);
        }
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    }
}

inline bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & static_cast<uintptr_t>(15)) == 0;
}

inline bool range_ok(uintptr_t base, size_t len, uintptr_t& end) {
    if (base > UINTPTR_MAX - static_cast<uintptr_t>(len)) return false;
    end = base + static_cast<uintptr_t>(len);
    return true;
}

inline bool disjoint(uintptr_t b1, uintptr_t e1, uintptr_t b2, uintptr_t e2) {
    return e1 <= b2 || e2 <= b1;
}

} // namespace

extern "C" int copy_four_segments(void* p0, void* p1, void* p2, void* p3,
                                  void* packed, size_t segment_bytes, int scatter,
                                  void* stream) {
    if (scatter != 0 && scatter != 1) return static_cast<int>(cudaErrorInvalidValue);
    if (segment_bytes == 0 || (segment_bytes % 16) != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (segment_bytes > (16u << 20)) return static_cast<int>(cudaErrorInvalidValue);
    if (p0 == nullptr || p1 == nullptr || p2 == nullptr || p3 == nullptr || packed == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    void* ptrs[5] = {p0, p1, p2, p3, packed};
    for (int i = 0; i < 5; ++i)
        if (!aligned16(ptrs[i])) return static_cast<int>(cudaErrorInvalidValue);

    const size_t total_bytes = 4 * segment_bytes; // segment_bytes <= 16MiB so no overflow
    uintptr_t bases[5], ends[5];
    for (int i = 0; i < 5; ++i) {
        size_t len = (i == 4) ? total_bytes : segment_bytes;
        if (!range_ok(reinterpret_cast<uintptr_t>(ptrs[i]), len, ends[i]))
            return static_cast<int>(cudaErrorInvalidValue);
        bases[i] = reinterpret_cast<uintptr_t>(ptrs[i]);
    }
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (!disjoint(bases[i], ends[i], bases[j], ends[j]))
                return static_cast<int>(cudaErrorInvalidValue);

    const size_t seg_vecs = segment_bytes / sizeof(uint4);
    const size_t total_vecs = seg_vecs * 4;
    const int threads = 256;
    size_t blocks = (total_vecs + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    four_segment_kernel<<<static_cast<int>(blocks), threads, 0, s>>>(
        static_cast<unsigned char*>(p0), static_cast<unsigned char*>(p1),
        static_cast<unsigned char*>(p2), static_cast<unsigned char*>(p3),
        static_cast<unsigned char*>(packed), seg_vecs, scatter, total_vecs);
    return static_cast<int>(cudaGetLastError());
}
