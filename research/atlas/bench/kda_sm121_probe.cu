// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glm53_kda_decode.cu"

namespace {

constexpr int DIM = 128;
constexpr int HEADS = 3;
constexpr int STEPS = 32;

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(2);
  }
}

float sigmoid(float x) {
  if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
  const float e = std::exp(x);
  return e / (1.0f + e);
}

float sample(int i, float scale) {
  const unsigned int x = static_cast<unsigned int>(i) * 1664525u + 1013904223u;
  return scale * (static_cast<float>(x & 0xffffu) / 32767.5f - 1.0f);
}

void cpu_step(const std::vector<__nv_bfloat16>& query,
              const std::vector<__nv_bfloat16>& key,
              const std::vector<__nv_bfloat16>& value,
              const std::vector<float>& log_decay,
              const std::vector<float>& beta,
              std::vector<float>* state,
              std::vector<float>* output) {
  for (int head = 0; head < HEADS; ++head) {
    const int vector_base = head * DIM;
    const int state_base = head * DIM * DIM;
    float q_sq = 0.0f;
    float k_sq = 0.0f;
    for (int k = 0; k < DIM; ++k) {
      const float q = __bfloat162float(query[vector_base + k]);
      const float key_value = __bfloat162float(key[vector_base + k]);
      q_sq += q * q;
      k_sq += key_value * key_value;
    }
    const float inv_q = 1.0f / std::sqrt(q_sq + 1.0e-6f) / std::sqrt(128.0f);
    const float inv_k = 1.0f / std::sqrt(k_sq + 1.0e-6f);
    std::vector<float> q_norm(DIM);
    std::vector<float> k_norm(DIM);
    for (int k = 0; k < DIM; ++k) {
      q_norm[k] = __bfloat162float(query[vector_base + k]) * inv_q;
      k_norm[k] = __bfloat162float(key[vector_base + k]) * inv_k;
    }

    for (int k = 0; k < DIM; ++k) {
      const float decay = std::exp(log_decay[vector_base + k]);
      for (int v = 0; v < DIM; ++v) {
        (*state)[state_base + v * DIM + k] *= decay;
      }
    }
    std::vector<float> delta(DIM);
    for (int v = 0; v < DIM; ++v) {
      float memory = 0.0f;
      for (int k = 0; k < DIM; ++k) {
        memory += k_norm[k] * (*state)[state_base + v * DIM + k];
      }
      delta[v] = beta[head] *
                 (__bfloat162float(value[vector_base + v]) - memory);
    }
    for (int k = 0; k < DIM; ++k) {
      for (int v = 0; v < DIM; ++v) {
        (*state)[state_base + v * DIM + k] += k_norm[k] * delta[v];
      }
    }
    for (int v = 0; v < DIM; ++v) {
      float result = 0.0f;
      for (int k = 0; k < DIM; ++k) {
        result += q_norm[k] * (*state)[state_base + v * DIM + k];
      }
      (*output)[vector_base + v] = result;
    }
  }
}

}  // namespace

int main() {
  const int vectors = HEADS * DIM;
  const int states = HEADS * DIM * DIM;
  std::vector<__nv_bfloat16> query(vectors), key(vectors), value(vectors);
  std::vector<float> log_decay(vectors), beta(HEADS), state(states);
  for (int i = 0; i < states; ++i) state[i] = sample(i + 2003, 0.08f);

  std::vector<float> expected_state = state;
  std::vector<float> expected_output(vectors);

  __nv_bfloat16 *d_query, *d_key, *d_value, *d_output;
  float *d_decay, *d_beta, *d_state;
  cuda_check(cudaMalloc(&d_query, vectors * sizeof(*d_query)), "cudaMalloc query");
  cuda_check(cudaMalloc(&d_key, vectors * sizeof(*d_key)), "cudaMalloc key");
  cuda_check(cudaMalloc(&d_value, vectors * sizeof(*d_value)), "cudaMalloc value");
  cuda_check(cudaMalloc(&d_decay, vectors * sizeof(*d_decay)), "cudaMalloc decay");
  cuda_check(cudaMalloc(&d_beta, HEADS * sizeof(*d_beta)), "cudaMalloc beta");
  cuda_check(cudaMalloc(&d_state, states * sizeof(*d_state)), "cudaMalloc state");
  cuda_check(cudaMalloc(&d_output, vectors * sizeof(*d_output)), "cudaMalloc output");
  cuda_check(cudaMemcpy(d_state, state.data(), states * sizeof(*d_state), cudaMemcpyHostToDevice), "copy state");

  std::vector<float> actual_state(states);
  std::vector<__nv_bfloat16> actual_output(vectors);
  float output_max_abs = 0.0f;
  for (int step = 0; step < STEPS; ++step) {
    const int token_seed = step * 4099;
    for (int i = 0; i < vectors; ++i) {
      query[i] = __float2bfloat16_rn(sample(token_seed + i + 1, 0.75f));
      key[i] = __float2bfloat16_rn(sample(token_seed + i + 401, 0.65f));
      value[i] = __float2bfloat16_rn(sample(token_seed + i + 809, 1.25f));
      const float forget = sample(token_seed + i + 1201, 0.4f);
      const float dt_bias = sample(token_seed + i + 1601, 0.2f);
      const float a_log = -0.3f + 0.05f * static_cast<float>(i / DIM);
      log_decay[i] =
          -5.0f * sigmoid(std::exp(a_log) * (forget + dt_bias));
    }
    for (int head = 0; head < HEADS; ++head) {
      beta[head] = sigmoid(-0.4f + 0.3f * static_cast<float>(head) +
                           0.01f * static_cast<float>(step));
    }
    cpu_step(query, key, value, log_decay, beta, &expected_state,
             &expected_output);

    cuda_check(cudaMemcpy(d_query, query.data(), vectors * sizeof(*d_query),
                          cudaMemcpyHostToDevice),
               "copy query");
    cuda_check(cudaMemcpy(d_key, key.data(), vectors * sizeof(*d_key),
                          cudaMemcpyHostToDevice),
               "copy key");
    cuda_check(cudaMemcpy(d_value, value.data(), vectors * sizeof(*d_value),
                          cudaMemcpyHostToDevice),
               "copy value");
    cuda_check(cudaMemcpy(d_decay, log_decay.data(),
                          vectors * sizeof(*d_decay), cudaMemcpyHostToDevice),
               "copy decay");
    cuda_check(cudaMemcpy(d_beta, beta.data(), HEADS * sizeof(*d_beta),
                          cudaMemcpyHostToDevice),
               "copy beta");

    glm53_kda_decode<<<HEADS, 256>>>(d_query, d_key, d_value, d_decay, d_beta,
                                    d_state, d_output, HEADS);
    cuda_check(cudaGetLastError(), "glm53_kda_decode launch");
    cuda_check(cudaDeviceSynchronize(), "glm53_kda_decode synchronize");
    cuda_check(cudaMemcpy(actual_output.data(), d_output,
                          vectors * sizeof(*d_output), cudaMemcpyDeviceToHost),
               "copy output back");
    for (int i = 0; i < vectors; ++i) {
      output_max_abs = std::max(
          output_max_abs,
          std::fabs(__bfloat162float(actual_output[i]) - expected_output[i]));
    }
  }

  cuda_check(cudaMemcpy(actual_state.data(), d_state, states * sizeof(*d_state), cudaMemcpyDeviceToHost), "copy state back");

  float state_max_abs = 0.0f;
  for (int i = 0; i < states; ++i) {
    state_max_abs = std::max(state_max_abs, std::fabs(actual_state[i] - expected_state[i]));
  }
  std::printf(
      "heads=%d steps=%d state_max_abs=%.9g output_bf16_max_abs=%.9g\n",
      HEADS, STEPS, state_max_abs, output_max_abs);

  cudaFree(d_output);
  cudaFree(d_state);
  cudaFree(d_beta);
  cudaFree(d_decay);
  cudaFree(d_value);
  cudaFree(d_key);
  cudaFree(d_query);
  return state_max_abs <= 2.0e-5f && output_max_abs <= 2.0e-3f ? 0 : 1;
}
