// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glm53_dsa_projection.cu"

namespace {

constexpr int kHeads = 64;
constexpr int kQk = 256;
constexpr int kLatent = 512;
constexpr int kRowsPerHead = 512;
constexpr int kTokens = 3;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(2);
  }
}

template <typename T>
T* managed(size_t count) {
  T* pointer = nullptr;
  check(cudaMallocManaged(&pointer, count * sizeof(T)), "cudaMallocManaged");
  std::fill(pointer, pointer + count, T{});
  return pointer;
}

float sample(int seed, float scale) {
  const unsigned int value = static_cast<unsigned int>(seed) * 1664525u +
                             1013904223u;
  return scale * (static_cast<float>(value & 0xffffu) / 32767.5f - 1.0f);
}

float elapsed(cudaEvent_t begin, cudaEvent_t end) {
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
  return milliseconds;
}

}  // namespace

int main() {
  const size_t weight_rows = kHeads * kRowsPerHead;
  auto* weight = managed<unsigned char>(weight_rows * kLatent);
  auto* scale = managed<float>((weight_rows / 128) * (kLatent / 128));
  for (size_t index = 0; index < (weight_rows / 128) * 4; ++index) {
    scale[index] = 0.5f + static_cast<float>(index % 11) * 0.0625f;
  }
  for (int head = 0; head < kHeads; ++head) {
    for (int channel = 0; channel < kQk; ++channel) {
      const size_t key_row = head * kRowsPerHead + channel;
      const size_t value_row = head * kRowsPerHead + kQk + channel;
      weight[key_row * kLatent + channel] = 0x38;    // E4M3 1.0
      weight[value_row * kLatent + channel] = 0xb8;  // E4M3 -1.0
    }
  }

  auto* query = managed<__nv_bfloat16>(kTokens * kHeads * kQk);
  auto* absorbed = managed<__nv_bfloat16>(kTokens * kHeads * kLatent);
  auto* latent = managed<__nv_bfloat16>(kTokens * kHeads * kLatent);
  auto* expanded = managed<__nv_bfloat16>(kTokens * kHeads * kQk);
  for (int index = 0; index < kTokens * kHeads * kQk; ++index) {
    query[index] = __float2bfloat16_rn(sample(index + 11, 0.5f));
  }
  for (int index = 0; index < kTokens * kHeads * kLatent; ++index) {
    latent[index] = __float2bfloat16_rn(sample(index + 71, 0.25f));
  }

  glm53_dsa_absorb_query_fp8<<<dim3(kHeads, kTokens), 256>>>(
      query, weight, scale, absorbed, kTokens);
  glm53_dsa_expand_value_fp8<<<dim3(kHeads, kTokens), 256>>>(
      latent, weight, scale, expanded, kTokens);
  check(cudaGetLastError(), "absorbed MLA projection launch");
  check(cudaDeviceSynchronize(), "absorbed MLA projection synchronize");

  float absorb_max_abs = 0.0f;
  float expand_max_abs = 0.0f;
  for (int token = 0; token < kTokens; ++token) {
    for (int head = 0; head < kHeads; ++head) {
      for (int channel = 0; channel < kQk; ++channel) {
        const int weight_row = head * kRowsPerHead + channel;
        const float key_scale = scale[(weight_row / 128) * 4 + channel / 128];
        const float expected_absorb = __bfloat162float(__float2bfloat16_rn(
            __bfloat162float(query[(token * kHeads + head) * kQk + channel]) *
            key_scale));
        absorb_max_abs = std::max(
            absorb_max_abs,
            std::fabs(__bfloat162float(
                          absorbed[(token * kHeads + head) * kLatent + channel]) -
                      expected_absorb));
        const int value_row = head * kRowsPerHead + kQk + channel;
        const float value_scale = scale[(value_row / 128) * 4 + channel / 128];
        const float expected_expand = __bfloat162float(__float2bfloat16_rn(
            -__bfloat162float(latent[(token * kHeads + head) * kLatent + channel]) *
            value_scale));
        expand_max_abs = std::max(
            expand_max_abs,
            std::fabs(__bfloat162float(
                          expanded[(token * kHeads + head) * kQk + channel]) -
                      expected_expand));
      }
      for (int channel = kQk; channel < kLatent; ++channel) {
        absorb_max_abs = std::max(
            absorb_max_abs,
            std::fabs(__bfloat162float(
                absorbed[(token * kHeads + head) * kLatent + channel])));
      }
    }
  }

  auto* norm_input = managed<__nv_bfloat16>(kTokens * 128);
  auto* norm_weight = managed<__nv_bfloat16>(128);
  auto* norm_bias = managed<__nv_bfloat16>(128);
  auto* norm_output = managed<__nv_bfloat16>(kTokens * 128);
  for (int channel = 0; channel < 128; ++channel) {
    norm_weight[channel] = __float2bfloat16_rn(0.8f + channel * 0.001f);
    norm_bias[channel] = __float2bfloat16_rn((channel - 64) * 0.0005f);
    for (int token = 0; token < kTokens; ++token) {
      norm_input[token * 128 + channel] =
          __float2bfloat16_rn(sample(token * 128 + channel + 901, 0.8f));
    }
  }
  glm53_dsa_index_layernorm<<<kTokens, 128>>>(
      norm_input, norm_weight, norm_bias, norm_output, kTokens, 1.0e-6f);
  check(cudaGetLastError(), "index layernorm launch");
  check(cudaDeviceSynchronize(), "index layernorm synchronize");
  float norm_max_abs = 0.0f;
  for (int token = 0; token < kTokens; ++token) {
    float mean = 0.0f;
    for (int channel = 0; channel < 128; ++channel) {
      mean += __bfloat162float(norm_input[token * 128 + channel]);
    }
    mean /= 128.0f;
    float variance = 0.0f;
    for (int channel = 0; channel < 128; ++channel) {
      const float delta =
          __bfloat162float(norm_input[token * 128 + channel]) - mean;
      variance += delta * delta;
    }
    const float inverse = 1.0f / std::sqrt(variance / 128.0f + 1.0e-6f);
    for (int channel = 0; channel < 128; ++channel) {
      const float expected = __bfloat162float(__float2bfloat16_rn(
          (__bfloat162float(norm_input[token * 128 + channel]) - mean) * inverse *
              __bfloat162float(norm_weight[channel]) +
          __bfloat162float(norm_bias[channel])));
      norm_max_abs = std::max(
          norm_max_abs,
          std::fabs(__bfloat162float(norm_output[token * 128 + channel]) - expected));
    }
  }

  auto* slot0 = managed<__nv_bfloat16>(16 * kLatent);
  auto* slot1 = managed<__nv_bfloat16>(16 * kLatent);
  auto* state_ptrs = managed<unsigned long long>(10);
  state_ptrs[0] = reinterpret_cast<unsigned long long>(slot1);
  state_ptrs[5] = reinterpret_cast<unsigned long long>(slot0);
  auto* packed = managed<__nv_bfloat16>(kTokens * kLatent);
  auto* cu = managed<int>(3);
  auto* positions = managed<int>(kTokens);
  auto* valid = managed<unsigned char>(kTokens);
  cu[0] = 0;
  cu[1] = 1;
  cu[2] = 3;
  positions[0] = 7;
  positions[1] = 2;
  positions[2] = 9;
  valid[0] = 1;
  valid[1] = 0;
  valid[2] = 1;
  for (int index = 0; index < kTokens * kLatent; ++index) {
    packed[index] = __float2bfloat16_rn(sample(index + 1201, 0.3f));
  }
  glm53_dsa_latent_append<<<kTokens, 256>>>(
      packed, cu, positions, valid, state_ptrs, kTokens, 2, 16);
  check(cudaGetLastError(), "latent append launch");
  check(cudaDeviceSynchronize(), "latent append synchronize");
  bool latent_exact = true;
  for (int channel = 0; channel < kLatent; ++channel) {
    latent_exact &= slot1[7 * kLatent + channel] == packed[channel];
    latent_exact &= slot0[2 * kLatent + channel] == __float2bfloat16_rn(0.0f);
    latent_exact &= slot0[9 * kLatent + channel] == packed[2 * kLatent + channel];
  }

  cudaEvent_t begin;
  cudaEvent_t end;
  check(cudaEventCreate(&begin), "cudaEventCreate begin");
  check(cudaEventCreate(&end), "cudaEventCreate end");
  constexpr int iterations = 50;
  check(cudaEventRecord(begin), "cudaEventRecord begin");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    glm53_dsa_absorb_query_fp8<<<dim3(kHeads, kTokens), 256>>>(
        query, weight, scale, absorbed, kTokens);
    glm53_dsa_expand_value_fp8<<<dim3(kHeads, kTokens), 256>>>(
        latent, weight, scale, expanded, kTokens);
  }
  check(cudaEventRecord(end), "cudaEventRecord end");
  check(cudaEventSynchronize(end), "cudaEventSynchronize end");
  const float projection_us = elapsed(begin, end) * 1000.0f / iterations;

  std::printf(
      "absorb_max_abs=%.9g expand_max_abs=%.9g norm_max_abs=%.9g "
      "latent_exact=%s absorb_plus_expand_us=%.3f tokens=%d\n",
      absorb_max_abs, expand_max_abs, norm_max_abs,
      latent_exact ? "true" : "false", projection_us, kTokens);
  return absorb_max_abs == 0.0f && expand_max_abs == 0.0f &&
                 norm_max_abs <= 0.0078125f && latent_exact
             ? 0
             : 1;
}
