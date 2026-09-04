// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "glm53_kda_prefill.cu"

namespace {

constexpr int DIM = 128;
constexpr int HEADS = 3;
constexpr int SEQUENCES = 2;
constexpr int CAPACITY = 4;
constexpr int SLOTS[SEQUENCES] = {3, 1};
constexpr int LENGTHS[SEQUENCES] = {5, 18};
constexpr int TOKENS = LENGTHS[0] + LENGTHS[1];

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(2);
  }
}

float sample(int index, float scale) {
  const unsigned int value = static_cast<unsigned int>(index) * 1664525u +
                             1013904223u;
  return scale * (static_cast<float>(value & 0xffffu) / 32767.5f - 1.0f);
}

float host_sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

unsigned int bf16_ulp(__nv_bfloat16 left, __nv_bfloat16 right) {
  std::uint16_t left_bits = 0;
  std::uint16_t right_bits = 0;
  std::memcpy(&left_bits, &left, sizeof(left_bits));
  std::memcpy(&right_bits, &right, sizeof(right_bits));
  const unsigned int left_ordered =
      (left_bits & 0x8000u) != 0 ? static_cast<std::uint16_t>(~left_bits)
                                 : left_bits | 0x8000u;
  const unsigned int right_ordered =
      (right_bits & 0x8000u) != 0 ? static_cast<std::uint16_t>(~right_bits)
                                  : right_bits | 0x8000u;
  return left_ordered > right_ordered ? left_ordered - right_ordered
                                      : right_ordered - left_ordered;
}

template <typename T>
T* allocate(std::size_t count) {
  T* pointer = nullptr;
  check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
  return pointer;
}

template <typename T>
void upload(T* device, const std::vector<T>& host) {
  check(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                   cudaMemcpyHostToDevice),
        "upload");
}

template <typename T>
void download(std::vector<T>* host, const T* device) {
  check(cudaMemcpy(host->data(), device, host->size() * sizeof(T),
                   cudaMemcpyDeviceToHost),
        "download");
}

float conv_step(float* history, const __nv_bfloat16* weight,
                __nv_bfloat16 input) {
  const float current = __bfloat162float(input);
  const float result = history[0] * __bfloat162float(weight[0]) +
                       history[1] * __bfloat162float(weight[1]) +
                       history[2] * __bfloat162float(weight[2]) +
                       current * __bfloat162float(weight[3]);
  history[0] = history[1];
  history[1] = history[2];
  history[2] = current;
  return result * host_sigmoid(result);
}

void reference_conv(std::vector<__nv_bfloat16>* query,
                    std::vector<__nv_bfloat16>* key,
                    std::vector<__nv_bfloat16>* value,
                    const std::vector<__nv_bfloat16>& query_weight,
                    const std::vector<__nv_bfloat16>& key_weight,
                    const std::vector<__nv_bfloat16>& value_weight,
                    std::vector<float>* query_history,
                    std::vector<float>* key_history,
                    std::vector<float>* value_history) {
  int token_start = 0;
  for (int sequence = 0; sequence < SEQUENCES; ++sequence) {
    for (int head = 0; head < HEADS; ++head) {
      for (int channel = 0; channel < DIM; ++channel) {
        const int weight = (head * DIM + channel) * 4;
        const int history =
            ((SLOTS[sequence] * HEADS + head) * DIM + channel) * 3;
        for (int token = token_start;
             token < token_start + LENGTHS[sequence]; ++token) {
          const int element = (token * HEADS + head) * DIM + channel;
          (*query)[element] = __float2bfloat16_rn(
              conv_step(&(*query_history)[history], &query_weight[weight],
                        (*query)[element]));
          (*key)[element] = __float2bfloat16_rn(
              conv_step(&(*key_history)[history], &key_weight[weight],
                        (*key)[element]));
          (*value)[element] = __float2bfloat16_rn(
              conv_step(&(*value_history)[history], &value_weight[weight],
                        (*value)[element]));
        }
      }
    }
    token_start += LENGTHS[sequence];
  }
}

}  // namespace

int main() {
  const int elements = TOKENS * HEADS * DIM;
  const int weights = HEADS * DIM * 4;
  const int history_elements = CAPACITY * HEADS * DIM * 3;
  std::vector<__nv_bfloat16> query(elements), key(elements), value(elements);
  std::vector<__nv_bfloat16> query_weight(weights), key_weight(weights);
  std::vector<__nv_bfloat16> value_weight(weights);
  std::vector<float> query_history(history_elements);
  std::vector<float> key_history(history_elements), value_history(history_elements);
  for (int index = 0; index < elements; ++index) {
    query[index] = __float2bfloat16_rn(sample(index + 1, 0.8f));
    key[index] = __float2bfloat16_rn(sample(index + 20001, 0.7f));
    value[index] = __float2bfloat16_rn(sample(index + 40001, 0.9f));
  }
  for (int index = 0; index < weights; ++index) {
    query_weight[index] = __float2bfloat16_rn(sample(index + 60001, 0.3f));
    key_weight[index] = __float2bfloat16_rn(sample(index + 70001, 0.3f));
    value_weight[index] = __float2bfloat16_rn(sample(index + 80001, 0.3f));
  }
  for (int index = 0; index < history_elements; ++index) {
    query_history[index] = sample(index + 90001, 0.05f);
    key_history[index] = sample(index + 100001, 0.05f);
    value_history[index] = sample(index + 110001, 0.05f);
  }
  auto expected_query = query;
  auto expected_key = key;
  auto expected_value = value;
  auto expected_query_history = query_history;
  auto expected_key_history = key_history;
  auto expected_value_history = value_history;
  reference_conv(&expected_query, &expected_key, &expected_value, query_weight,
                 key_weight, value_weight, &expected_query_history,
                 &expected_key_history, &expected_value_history);

  auto* d_query = allocate<__nv_bfloat16>(elements);
  auto* d_key = allocate<__nv_bfloat16>(elements);
  auto* d_value = allocate<__nv_bfloat16>(elements);
  auto* d_qw = allocate<__nv_bfloat16>(weights);
  auto* d_kw = allocate<__nv_bfloat16>(weights);
  auto* d_vw = allocate<__nv_bfloat16>(weights);
  auto* d_qh = allocate<float>(history_elements);
  auto* d_kh = allocate<float>(history_elements);
  auto* d_vh = allocate<float>(history_elements);
  auto* d_cu = allocate<std::int64_t>(SEQUENCES + 1);
  auto* d_ptrs = allocate<unsigned long long>(SEQUENCES * 4);
  upload(d_query, query);
  upload(d_key, key);
  upload(d_value, value);
  upload(d_qw, query_weight);
  upload(d_kw, key_weight);
  upload(d_vw, value_weight);
  upload(d_qh, query_history);
  upload(d_kh, key_history);
  upload(d_vh, value_history);
  upload(d_cu, std::vector<std::int64_t>{0, LENGTHS[0], TOKENS});
  std::vector<unsigned long long> pointers(SEQUENCES * 4);
  for (int sequence = 0; sequence < SEQUENCES; ++sequence) {
    const int offset = SLOTS[sequence] * HEADS * DIM * 3;
    pointers[sequence * 4 + 1] =
        reinterpret_cast<unsigned long long>(d_qh + offset);
    pointers[sequence * 4 + 2] =
        reinterpret_cast<unsigned long long>(d_kh + offset);
    pointers[sequence * 4 + 3] =
        reinterpret_cast<unsigned long long>(d_vh + offset);
  }
  upload(d_ptrs, pointers);
  glm53_kda_conv_silu_chunk<<<SEQUENCES * HEADS, DIM>>>(
      d_query, d_key, d_value, d_qw, d_kw, d_vw, d_cu, d_ptrs, SEQUENCES,
      HEADS);
  check(cudaDeviceSynchronize(), "conv synchronize");
  download(&query, d_query);
  download(&key, d_key);
  download(&value, d_value);
  download(&query_history, d_qh);
  download(&key_history, d_kh);
  download(&value_history, d_vh);
  unsigned int conv_max_ulp = 0;
  float history_max_abs = 0.0f;
  for (int index = 0; index < elements; ++index) {
    conv_max_ulp = std::max(conv_max_ulp, bf16_ulp(query[index], expected_query[index]));
    conv_max_ulp = std::max(conv_max_ulp, bf16_ulp(key[index], expected_key[index]));
    conv_max_ulp = std::max(conv_max_ulp, bf16_ulp(value[index], expected_value[index]));
  }
  for (int index = 0; index < history_elements; ++index) {
    history_max_abs = std::max(history_max_abs,
        std::fabs(query_history[index] - expected_query_history[index]));
    history_max_abs = std::max(history_max_abs,
        std::fabs(key_history[index] - expected_key_history[index]));
    history_max_abs = std::max(history_max_abs,
        std::fabs(value_history[index] - expected_value_history[index]));
  }

  const int beta_elements = TOKENS * HEADS;
  std::vector<__nv_bfloat16> beta(beta_elements), beta_ht(beta_elements);
  for (int index = 0; index < beta_elements; ++index) {
    beta[index] = __float2bfloat16_rn(sample(index + 120001, 0.6f));
  }
  auto* d_beta = allocate<__nv_bfloat16>(beta_elements);
  auto* d_beta_ht = allocate<__nv_bfloat16>(beta_elements);
  upload(d_beta, beta);
  glm53_kda_beta_transpose<<<(beta_elements + 255) / 256, 256>>>(
      d_beta, d_beta_ht, TOKENS, HEADS);
  check(cudaDeviceSynchronize(), "beta synchronize");
  download(&beta_ht, d_beta_ht);
  bool beta_exact = true;
  for (int token = 0; token < TOKENS; ++token) {
    for (int head = 0; head < HEADS; ++head) {
      beta_exact &= beta_ht[head * TOKENS + token] == beta[token * HEADS + head];
    }
  }

  std::vector<__nv_bfloat16> gate(elements), norm_weight(DIM), normalized(elements);
  std::vector<__nv_bfloat16> expected_normalized(elements);
  for (int index = 0; index < elements; ++index) {
    gate[index] = __float2bfloat16_rn(sample(index + 130001, 0.7f));
  }
  for (int channel = 0; channel < DIM; ++channel) {
    norm_weight[channel] = __float2bfloat16_rn(0.9f + sample(channel, 0.1f));
  }
  for (int row = 0; row < TOKENS * HEADS; ++row) {
    float reduction[DIM];
    for (int channel = 0; channel < DIM; ++channel) {
      const float core = __bfloat162float(query[row * DIM + channel]);
      reduction[channel] = core * core;
    }
    for (int stride = DIM / 2; stride != 0; stride >>= 1) {
      for (int channel = 0; channel < stride; ++channel) {
        reduction[channel] += reduction[channel + stride];
      }
    }
    const float inverse = 1.0f / std::sqrt(reduction[0] / DIM + 1.0e-6f);
    for (int channel = 0; channel < DIM; ++channel) {
      const int element = row * DIM + channel;
      expected_normalized[element] = __float2bfloat16_rn(
          __bfloat162float(query[element]) * inverse *
          __bfloat162float(norm_weight[channel]) *
          host_sigmoid(__bfloat162float(gate[element])));
    }
  }
  auto* d_gate = allocate<__nv_bfloat16>(elements);
  auto* d_norm = allocate<__nv_bfloat16>(DIM);
  auto* d_normalized = allocate<__nv_bfloat16>(elements);
  upload(d_gate, gate);
  upload(d_norm, norm_weight);
  glm53_kda_gated_norm_chunk<<<TOKENS * HEADS, DIM>>>(
      d_query, d_gate, d_norm, d_normalized, TOKENS * HEADS, 1.0e-6f);
  check(cudaDeviceSynchronize(), "norm synchronize");
  download(&normalized, d_normalized);
  unsigned int norm_max_ulp = 0;
  for (int index = 0; index < elements; ++index) {
    norm_max_ulp = std::max(
        norm_max_ulp, bf16_ulp(normalized[index], expected_normalized[index]));
  }

  std::printf(
      "tokens=%d heads=%d slots={%d,%d} conv_max_ulp=%u "
      "history_max_abs=%.9g beta_exact=%s norm_max_ulp=%u\n",
      TOKENS, HEADS, SLOTS[0], SLOTS[1], conv_max_ulp, history_max_abs,
      beta_exact ? "true" : "false", norm_max_ulp);
  return conv_max_ulp <= 2 && history_max_abs == 0.0f && beta_exact &&
                 norm_max_ulp <= 2
             ? 0
             : 1;
}
