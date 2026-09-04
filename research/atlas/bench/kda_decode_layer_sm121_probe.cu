// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glm53_kda_decode.cu"

namespace {

constexpr int DIM = 128;
constexpr int NUM_HEADS = 3;
constexpr int BATCH = 2;
constexpr int BATCH_HEADS = NUM_HEADS * BATCH;
constexpr int SLOT_COUNT = 4;
constexpr int SLOT_ORDER[BATCH] = {3, 1};
constexpr int STEPS = 32;
constexpr float NORM_EPS = 1.0e-6f;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(2);
  }
}

float sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

float bf16_round(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

unsigned int bf16_ulp_distance(__nv_bfloat16 lhs, __nv_bfloat16 rhs) {
  std::uint16_t lhs_bits = 0;
  std::uint16_t rhs_bits = 0;
  static_assert(sizeof(lhs_bits) == sizeof(lhs));
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  const unsigned int lhs_ordered =
      (lhs_bits & 0x8000u) != 0 ? static_cast<std::uint16_t>(~lhs_bits)
                                : static_cast<unsigned int>(lhs_bits | 0x8000u);
  const unsigned int rhs_ordered =
      (rhs_bits & 0x8000u) != 0 ? static_cast<std::uint16_t>(~rhs_bits)
                                : static_cast<unsigned int>(rhs_bits | 0x8000u);
  return lhs_ordered > rhs_ordered ? lhs_ordered - rhs_ordered
                                   : rhs_ordered - lhs_ordered;
}

float sample(int index, float scale) {
  const unsigned int x = static_cast<unsigned int>(index) * 1664525u +
                         1013904223u;
  return scale * (static_cast<float>(x & 0xffffu) / 32767.5f - 1.0f);
}

float conv_step(float* history, const __nv_bfloat16* weight,
                __nv_bfloat16 input) {
  const float current = __bfloat162float(input);
  const float conv = history[0] * __bfloat162float(weight[0]) +
                     history[1] * __bfloat162float(weight[1]) +
                     history[2] * __bfloat162float(weight[2]) +
                     current * __bfloat162float(weight[3]);
  history[0] = history[1];
  history[1] = history[2];
  history[2] = current;
  return bf16_round(conv * sigmoid(conv));
}

void cpu_step(
    const std::vector<__nv_bfloat16>& query,
    const std::vector<__nv_bfloat16>& key,
    const std::vector<__nv_bfloat16>& value,
    const std::vector<__nv_bfloat16>& query_weight,
    const std::vector<__nv_bfloat16>& key_weight,
    const std::vector<__nv_bfloat16>& value_weight,
    std::vector<float>* query_history,
    std::vector<float>* key_history,
    std::vector<float>* value_history,
    const std::vector<__nv_bfloat16>& forget,
    const std::vector<float>& dt_bias,
    const std::vector<float>& a_log,
    const std::vector<__nv_bfloat16>& beta_logit,
    const std::vector<__nv_bfloat16>& output_gate,
    const std::vector<__nv_bfloat16>& norm_weight,
    std::vector<float>* state,
    std::vector<__nv_bfloat16>* output) {
  for (int head = 0; head < BATCH_HEADS; ++head) {
    const int sequence = head / NUM_HEADS;
    const int model_head = head % NUM_HEADS;
    const int vector_base = head * DIM;
    const int model_base = model_head * DIM;
    const int physical_head = SLOT_ORDER[sequence] * NUM_HEADS + model_head;
    const int state_base = physical_head * DIM * DIM;
    std::vector<float> q(DIM), k(DIM), v(DIM), decay(DIM);
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    for (int channel = 0; channel < DIM; ++channel) {
      const int vector = vector_base + channel;
      const int model_vector = model_base + channel;
      const int history = physical_head * DIM * 3 + channel * 3;
      const int weight = model_vector * 4;
      q[channel] = conv_step(&(*query_history)[history],
                             &query_weight[weight], query[vector]);
      k[channel] = conv_step(&(*key_history)[history], &key_weight[weight],
                             key[vector]);
      v[channel] = conv_step(&(*value_history)[history],
                             &value_weight[weight], value[vector]);
      q_sq += q[channel] * q[channel];
      k_sq += k[channel] * k[channel];
      const float log_decay =
          -5.0f * sigmoid(std::exp(a_log[model_head]) *
                          (__bfloat162float(forget[vector]) +
                           dt_bias[model_vector]));
      decay[channel] = std::exp(log_decay);
    }
    const float inv_q = 1.0f / std::sqrt(q_sq + 1.0e-6f) /
                        std::sqrt(static_cast<float>(DIM));
    const float inv_k = 1.0f / std::sqrt(k_sq + 1.0e-6f);
    for (int channel = 0; channel < DIM; ++channel) {
      q[channel] *= inv_q;
      k[channel] *= inv_k;
    }
    const float beta = bf16_round(sigmoid(
        __bfloat162float(beta_logit[head])));
    std::vector<float> delta(DIM), core(DIM);
    for (int value_index = 0; value_index < DIM; ++value_index) {
      float memory = 0.0f;
      for (int key_index = 0; key_index < DIM; ++key_index) {
        const int index = state_base + value_index * DIM + key_index;
        (*state)[index] *= decay[key_index];
        memory += k[key_index] * (*state)[index];
      }
      delta[value_index] = beta * (v[value_index] - memory);
    }
    for (int key_index = 0; key_index < DIM; ++key_index) {
      for (int value_index = 0; value_index < DIM; ++value_index) {
        (*state)[state_base + value_index * DIM + key_index] +=
            k[key_index] * delta[value_index];
      }
    }
    float square_sum = 0.0f;
    for (int value_index = 0; value_index < DIM; ++value_index) {
      float result = 0.0f;
      for (int key_index = 0; key_index < DIM; ++key_index) {
        result += q[key_index] *
                  (*state)[state_base + value_index * DIM + key_index];
      }
      core[value_index] = bf16_round(result);
      square_sum += core[value_index] * core[value_index];
    }
    const float rms = 1.0f /
                      std::sqrt(square_sum / static_cast<float>(DIM) +
                                NORM_EPS);
    for (int value_index = 0; value_index < DIM; ++value_index) {
      const float gated =
          core[value_index] * rms *
          __bfloat162float(norm_weight[value_index]) *
          sigmoid(__bfloat162float(output_gate[vector_base + value_index]));
      (*output)[vector_base + value_index] = __float2bfloat16_rn(gated);
    }
  }
}

template <typename T>
T* allocate(size_t count) {
  T* pointer = nullptr;
  check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
  return pointer;
}

template <typename T>
void upload(T* device, const std::vector<T>& host) {
  check(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy upload");
}

template <typename T>
void download(std::vector<T>* host, const T* device) {
  check(cudaMemcpy(host->data(), device, host->size() * sizeof(T),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy download");
}

}  // namespace

int main() {
  const int vectors = BATCH_HEADS * DIM;
  const int model_vectors = NUM_HEADS * DIM;
  const int state_size = SLOT_COUNT * NUM_HEADS * DIM * DIM;
  const int history_size = SLOT_COUNT * NUM_HEADS * DIM * 3;
  std::vector<__nv_bfloat16> query(vectors), key(vectors), value(vectors);
  std::vector<__nv_bfloat16> forget(vectors), beta_logit(BATCH_HEADS);
  std::vector<__nv_bfloat16> output_gate(vectors), output(vectors);
  std::vector<__nv_bfloat16> expected_output(vectors);
  std::vector<__nv_bfloat16> query_weight(model_vectors * 4);
  std::vector<__nv_bfloat16> key_weight(model_vectors * 4);
  std::vector<__nv_bfloat16> value_weight(model_vectors * 4);
  std::vector<__nv_bfloat16> norm_weight(DIM);
  std::vector<float> dt_bias(model_vectors), a_log(NUM_HEADS);
  std::vector<float> query_history(history_size);
  std::vector<float> key_history(history_size), value_history(history_size);
  std::vector<float> state(state_size);
  for (int i = 0; i < model_vectors * 4; ++i) {
    query_weight[i] = __float2bfloat16_rn(sample(i + 1, 0.35f));
    key_weight[i] = __float2bfloat16_rn(sample(i + 2001, 0.3f));
    value_weight[i] = __float2bfloat16_rn(sample(i + 4001, 0.4f));
  }
  for (int i = 0; i < model_vectors; ++i) dt_bias[i] = sample(i + 6001, 0.2f);
  for (int i = 0; i < NUM_HEADS; ++i) a_log[i] = -0.4f + 0.1f * i;
  for (int i = 0; i < DIM; ++i) {
    norm_weight[i] = __float2bfloat16_rn(0.8f + sample(i + 7001, 0.2f));
  }
  for (int i = 0; i < history_size; ++i) {
    query_history[i] = sample(i + 8001, 0.06f);
    key_history[i] = sample(i + 10001, 0.06f);
    value_history[i] = sample(i + 12001, 0.06f);
  }
  for (int i = 0; i < state_size; ++i) state[i] = sample(i + 14001, 0.05f);

  auto expected_query_history = query_history;
  auto expected_key_history = key_history;
  auto expected_value_history = value_history;
  auto expected_state = state;

  auto* d_query = allocate<__nv_bfloat16>(vectors);
  auto* d_key = allocate<__nv_bfloat16>(vectors);
  auto* d_value = allocate<__nv_bfloat16>(vectors);
  auto* d_forget = allocate<__nv_bfloat16>(vectors);
  auto* d_beta = allocate<__nv_bfloat16>(BATCH_HEADS);
  auto* d_gate = allocate<__nv_bfloat16>(vectors);
  auto* d_output = allocate<__nv_bfloat16>(vectors);
  auto* d_qw = allocate<__nv_bfloat16>(query_weight.size());
  auto* d_kw = allocate<__nv_bfloat16>(key_weight.size());
  auto* d_vw = allocate<__nv_bfloat16>(value_weight.size());
  auto* d_norm = allocate<__nv_bfloat16>(DIM);
  auto* d_dt = allocate<float>(model_vectors);
  auto* d_a = allocate<float>(NUM_HEADS);
  auto* d_qh = allocate<float>(history_size);
  auto* d_kh = allocate<float>(history_size);
  auto* d_vh = allocate<float>(history_size);
  auto* d_state = allocate<float>(state_size);
  auto* d_state_ptrs = allocate<unsigned long long>(BATCH * 4);
  upload(d_qw, query_weight);
  upload(d_kw, key_weight);
  upload(d_vw, value_weight);
  upload(d_norm, norm_weight);
  upload(d_dt, dt_bias);
  upload(d_a, a_log);
  upload(d_qh, query_history);
  upload(d_kh, key_history);
  upload(d_vh, value_history);
  upload(d_state, state);
  std::vector<unsigned long long> state_ptrs(BATCH * 4);
  for (int sequence = 0; sequence < BATCH; ++sequence) {
    const int slot = SLOT_ORDER[sequence];
    state_ptrs[sequence * 4] = reinterpret_cast<unsigned long long>(
        d_state + slot * NUM_HEADS * DIM * DIM);
    state_ptrs[sequence * 4 + 1] = reinterpret_cast<unsigned long long>(
        d_qh + slot * NUM_HEADS * DIM * 3);
    state_ptrs[sequence * 4 + 2] = reinterpret_cast<unsigned long long>(
        d_kh + slot * NUM_HEADS * DIM * 3);
    state_ptrs[sequence * 4 + 3] = reinterpret_cast<unsigned long long>(
        d_vh + slot * NUM_HEADS * DIM * 3);
  }
  upload(d_state_ptrs, state_ptrs);

  float output_max_abs = 0.0f;
  unsigned int output_max_ulp = 0;
  for (int step = 0; step < STEPS; ++step) {
    const int seed = step * 7919;
    for (int i = 0; i < vectors; ++i) {
      query[i] = __float2bfloat16_rn(sample(seed + i + 1, 0.9f));
      key[i] = __float2bfloat16_rn(sample(seed + i + 1001, 0.8f));
      value[i] = __float2bfloat16_rn(sample(seed + i + 2001, 1.1f));
      forget[i] = __float2bfloat16_rn(sample(seed + i + 3001, 0.45f));
      output_gate[i] = __float2bfloat16_rn(sample(seed + i + 4001, 0.7f));
    }
    for (int i = 0; i < BATCH_HEADS; ++i) {
      beta_logit[i] = __float2bfloat16_rn(sample(seed + i + 5001, 0.6f));
    }
    cpu_step(query, key, value, query_weight, key_weight, value_weight,
             &expected_query_history, &expected_key_history,
             &expected_value_history, forget, dt_bias, a_log, beta_logit,
             output_gate, norm_weight, &expected_state, &expected_output);
    upload(d_query, query);
    upload(d_key, key);
    upload(d_value, value);
    upload(d_forget, forget);
    upload(d_beta, beta_logit);
    upload(d_gate, output_gate);
    glm53_kda_decode_fused_conv_gate_norm<<<BATCH_HEADS, 256>>>(
        d_query, d_key, d_value, d_qw, d_kw, d_vw, d_state_ptrs,
        d_forget, d_dt, d_a, d_beta, d_gate, d_norm, d_output,
        BATCH_HEADS, NUM_HEADS, NORM_EPS);
    check(cudaGetLastError(), "fused KDA launch");
    check(cudaDeviceSynchronize(), "fused KDA synchronize");
    download(&output, d_output);
    for (int i = 0; i < vectors; ++i) {
      output_max_abs = std::max(
          output_max_abs,
          std::fabs(__bfloat162float(output[i]) -
                    __bfloat162float(expected_output[i])));
      output_max_ulp =
          std::max(output_max_ulp,
                   bf16_ulp_distance(output[i], expected_output[i]));
    }
  }

  download(&state, d_state);
  download(&query_history, d_qh);
  download(&key_history, d_kh);
  download(&value_history, d_vh);
  float state_max_abs = 0.0f;
  float history_max_abs = 0.0f;
  for (int i = 0; i < state_size; ++i) {
    state_max_abs =
        std::max(state_max_abs, std::fabs(state[i] - expected_state[i]));
  }
  for (int i = 0; i < history_size; ++i) {
    history_max_abs = std::max(
        history_max_abs,
        std::fabs(query_history[i] - expected_query_history[i]));
    history_max_abs = std::max(
        history_max_abs, std::fabs(key_history[i] - expected_key_history[i]));
    history_max_abs = std::max(
        history_max_abs,
        std::fabs(value_history[i] - expected_value_history[i]));
  }
  std::printf(
      "batch=%d heads=%d slots={%d,%d} steps=%d state_max_abs=%.9g "
      "history_max_abs=%.9g output_bf16_max_abs=%.9g "
      "output_bf16_max_ulp=%u\n",
      BATCH, NUM_HEADS, SLOT_ORDER[0], SLOT_ORDER[1], STEPS, state_max_abs,
      history_max_abs,
      output_max_abs, output_max_ulp);
  // The reference uses host libm while the production kernel follows the
  // CUDA/PyTorch rsqrtf and sigmoid path. Their final BF16 result may differ
  // by two representable values even when recurrent FP32 state agrees.
  return state_max_abs <= 3.0e-5f && history_max_abs == 0.0f &&
                 output_max_ulp <= 2
             ? 0
             : 1;
}
