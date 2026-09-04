// SPDX-License-Identifier: AGPL-3.0-only

// Checkpoint-free GB10 microbenchmark for the GLM-5.3 sparse-MLA target
// primitive. It compares Atlas's fixed 2,047-entry scan with an identical
// kernel whose loop ends at the request's actual visible-token count.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr unsigned int kHeads = 32;
constexpr unsigned int kMlaDim = 512;
constexpr unsigned int kWarps = 8;
constexpr unsigned int kCapacity = 2051;

#define CUDA_OK(call)                                                         \
  do {                                                                        \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                   cudaGetErrorString(status));                                \
      std::exit(1);                                                            \
    }                                                                         \
  } while (0)

__device__ __forceinline__ float warp_sum(float value) {
  for (unsigned int offset = 16; offset != 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <bool kBounded>
__global__ void sparse_mla(const __nv_bfloat16* __restrict__ query,
                           const __nv_bfloat16* __restrict__ latent,
                           const int* __restrict__ selected,
                           const unsigned int* __restrict__ valid_counts,
                           __nv_bfloat16* __restrict__ output,
                           unsigned int num_sequences) {
  const unsigned int head = blockIdx.x;
  const unsigned int sequence = blockIdx.y;
  const unsigned int tid = threadIdx.x;
  const unsigned int warp = tid >> 5;
  const unsigned int lane = tid & 31;
  if (head >= kHeads || sequence >= num_sequences) return;

  const unsigned int channel_base = lane * (kMlaDim / 32);
  const unsigned long long query_base =
      (static_cast<unsigned long long>(sequence) * kHeads + head) * kMlaDim;
  const unsigned long long selected_base =
      static_cast<unsigned long long>(sequence) * kCapacity;
  const unsigned int limit = kBounded ? valid_counts[sequence] : kCapacity;
  float maximum = -FLT_MAX;
  float denominator = 0.0f;
  float accumulator[kMlaDim / 32] = {};
  for (unsigned int rank = warp; rank < limit; rank += kWarps) {
    const int position = selected[selected_base + rank];
    if (position < 0) continue;
    const unsigned long long latent_base =
        (static_cast<unsigned long long>(sequence) * kCapacity + position) *
            kMlaDim +
        channel_base;
    float dot = 0.0f;
#pragma unroll
    for (unsigned int part = 0; part < kMlaDim / 32; ++part) {
      dot += __bfloat162float(query[query_base + channel_base + part]) *
             __bfloat162float(latent[latent_base + part]);
    }
    dot = __shfl_sync(0xffffffff, warp_sum(dot), 0);
    const float score = dot * 0.0625f;
    const float next_maximum = fmaxf(maximum, score);
    const float old_scale = expf(maximum - next_maximum);
    const float new_scale = expf(score - next_maximum);
    denominator = denominator * old_scale + new_scale;
#pragma unroll
    for (unsigned int part = 0; part < kMlaDim / 32; ++part) {
      accumulator[part] = accumulator[part] * old_scale +
                          new_scale * __bfloat162float(
                                          latent[latent_base + part]);
    }
    maximum = next_maximum;
  }

  __shared__ float warp_maximum[kWarps];
  __shared__ float warp_denominator[kWarps];
  __shared__ float warp_output[kWarps][kMlaDim];
  if (lane == 0) {
    warp_maximum[warp] = maximum;
    warp_denominator[warp] = denominator;
  }
#pragma unroll
  for (unsigned int part = 0; part < kMlaDim / 32; ++part) {
    warp_output[warp][channel_base + part] = accumulator[part];
  }
  __syncthreads();
  for (unsigned int stride = kWarps / 2; stride != 0; stride >>= 1) {
    if (warp < stride) {
      const unsigned int other = warp + stride;
      const float merged_maximum =
          fmaxf(warp_maximum[warp], warp_maximum[other]);
      const float left_scale = expf(warp_maximum[warp] - merged_maximum);
      const float right_scale = expf(warp_maximum[other] - merged_maximum);
      if (lane == 0) {
        warp_denominator[warp] = warp_denominator[warp] * left_scale +
                                 warp_denominator[other] * right_scale;
        warp_maximum[warp] = merged_maximum;
      }
#pragma unroll
      for (unsigned int part = 0; part < kMlaDim / 32; ++part) {
        const unsigned int channel = channel_base + part;
        warp_output[warp][channel] =
            warp_output[warp][channel] * left_scale +
            warp_output[other][channel] * right_scale;
      }
    }
    __syncthreads();
  }
  if (warp == 0) {
    const float inverse = warp_denominator[0] > 0.0f
                              ? 1.0f / warp_denominator[0]
                              : 0.0f;
#pragma unroll
    for (unsigned int part = 0; part < kMlaDim / 32; ++part) {
      output[query_base + channel_base + part] = __float2bfloat16_rn(
          warp_output[0][channel_base + part] * inverse);
    }
  }
}

template <bool kBounded>
float time_kernel(const __nv_bfloat16* query, const __nv_bfloat16* latent,
                  const int* selected, const unsigned int* valid_counts,
                  __nv_bfloat16* output, unsigned int num_sequences,
                  int iterations) {
  for (int i = 0; i < 20; ++i) {
    sparse_mla<kBounded><<<dim3(kHeads, num_sequences), 256>>>(
        query, latent, selected, valid_counts, output, num_sequences);
  }
  CUDA_OK(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  CUDA_OK(cudaEventCreate(&begin));
  CUDA_OK(cudaEventCreate(&end));
  CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) {
    sparse_mla<kBounded><<<dim3(kHeads, num_sequences), 256>>>(
        query, latent, selected, valid_counts, output, num_sequences);
  }
  CUDA_OK(cudaEventRecord(end));
  CUDA_OK(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CUDA_OK(cudaEventElapsedTime(&milliseconds, begin, end));
  CUDA_OK(cudaEventDestroy(begin));
  CUDA_OK(cudaEventDestroy(end));
  return milliseconds * 1000.0f / static_cast<float>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 200;
  constexpr unsigned int kMaxSequences = 4;
  const size_t query_elements = kMaxSequences * kHeads * kMlaDim;
  const size_t latent_elements = kMaxSequences * kCapacity * kMlaDim;
  const size_t selected_elements = kMaxSequences * kCapacity;
  std::vector<__nv_bfloat16> query_host(query_elements);
  std::vector<__nv_bfloat16> latent_host(latent_elements);
  for (size_t i = 0; i < query_elements; ++i) {
    query_host[i] =
        __float2bfloat16(static_cast<float>(static_cast<int>(i % 31) - 15) /
                         64.0f);
  }
  for (size_t i = 0; i < latent_elements; ++i) {
    latent_host[i] =
        __float2bfloat16(static_cast<float>(static_cast<int>(i % 29) - 14) /
                         64.0f);
  }

  __nv_bfloat16 *query, *latent, *fixed_output, *bounded_output;
  int* selected;
  unsigned int* valid_counts;
  CUDA_OK(cudaMalloc(&query, query_elements * sizeof(*query)));
  CUDA_OK(cudaMalloc(&latent, latent_elements * sizeof(*latent)));
  CUDA_OK(cudaMalloc(&selected, selected_elements * sizeof(*selected)));
  CUDA_OK(cudaMalloc(&valid_counts, kMaxSequences * sizeof(*valid_counts)));
  CUDA_OK(cudaMalloc(&fixed_output, query_elements * sizeof(*fixed_output)));
  CUDA_OK(cudaMalloc(&bounded_output, query_elements * sizeof(*bounded_output)));
  CUDA_OK(cudaMemcpy(query, query_host.data(), query_elements * sizeof(*query),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(latent, latent_host.data(),
                     latent_elements * sizeof(*latent),
                     cudaMemcpyHostToDevice));

  std::printf("sequences,visible,fixed_us,bounded_us,speedup,bit_exact\n");
  for (unsigned int sequences : {1u, 2u, 4u}) {
    for (unsigned int visible : {32u, 64u, 128u, 512u, 2051u}) {
      std::vector<int> selected_host(selected_elements, -1);
      std::vector<unsigned int> counts_host(kMaxSequences, visible);
      for (unsigned int sequence = 0; sequence < kMaxSequences; ++sequence) {
        for (unsigned int i = 0; i < visible; ++i) {
          selected_host[sequence * kCapacity + i] = static_cast<int>(i);
        }
      }
      CUDA_OK(cudaMemcpy(selected, selected_host.data(),
                         selected_elements * sizeof(*selected),
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(valid_counts, counts_host.data(),
                         kMaxSequences * sizeof(*valid_counts),
                         cudaMemcpyHostToDevice));
      const float fixed = time_kernel<false>(
          query, latent, selected, valid_counts, fixed_output, sequences,
          iterations);
      const float bounded = time_kernel<true>(
          query, latent, selected, valid_counts, bounded_output, sequences,
          iterations);
      std::vector<__nv_bfloat16> fixed_host(sequences * kHeads * kMlaDim);
      std::vector<__nv_bfloat16> bounded_host(fixed_host.size());
      CUDA_OK(cudaMemcpy(fixed_host.data(), fixed_output,
                         fixed_host.size() * sizeof(*fixed_output),
                         cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(bounded_host.data(), bounded_output,
                         bounded_host.size() * sizeof(*bounded_output),
                         cudaMemcpyDeviceToHost));
      const bool exact = std::memcmp(fixed_host.data(), bounded_host.data(),
                                     fixed_host.size() * sizeof(*fixed_output)) ==
                         0;
      std::printf("%u,%u,%.3f,%.3f,%.3f,%s\n", sequences, visible, fixed,
                  bounded, fixed / bounded, exact ? "yes" : "no");
    }
  }
  return 0;
}
