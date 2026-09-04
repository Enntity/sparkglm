// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "glm53_dsa_index.cu"

namespace {

constexpr int DIM = 128;
constexpr int HEADS = 32;
constexpr int BATCH = 2;
constexpr int MAX_POOLS = 520;
constexpr int OUTPUT_COUNT = 2051;
constexpr int PROBE_MLA_HEADS = 3;
constexpr int PROBE_LATENT = 512;

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

float bf16_round_host(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

float sample(int seed, float scale) {
  const unsigned int value = static_cast<unsigned int>(seed) * 1664525u +
                             1013904223u;
  return scale * (static_cast<float>(value & 0xffffu) / 32767.5f - 1.0f);
}

struct Sequence {
  std::vector<__nv_bfloat16> keys;
  std::vector<__nv_bfloat16> gates;
  std::vector<int> positions;
  std::vector<unsigned char> valid;

  int size() const { return static_cast<int>(positions.size()); }
};

struct Slot {
  __nv_bfloat16* latent = managed<__nv_bfloat16>((MAX_POOLS * 4 + 16) * 512);
  __nv_bfloat16* pooled = managed<__nv_bfloat16>(MAX_POOLS * DIM);
  __nv_bfloat16* tail_keys = managed<__nv_bfloat16>(3 * DIM);
  __nv_bfloat16* tail_gates = managed<__nv_bfloat16>(3 * DIM);
  int* metadata = managed<int>(4);
};

void set_state_ptrs(unsigned long long* pointers,
                    const std::vector<Slot>& slots,
                    int first,
                    int second) {
  const int order[BATCH] = {first, second};
  for (int sequence = 0; sequence < BATCH; ++sequence) {
    const Slot& slot = slots[order[sequence]];
    pointers[sequence * 5] =
        reinterpret_cast<unsigned long long>(slot.latent);
    pointers[sequence * 5 + 1] =
        reinterpret_cast<unsigned long long>(slot.pooled);
    pointers[sequence * 5 + 2] =
        reinterpret_cast<unsigned long long>(slot.tail_keys);
    pointers[sequence * 5 + 3] =
        reinterpret_cast<unsigned long long>(slot.tail_gates);
    pointers[sequence * 5 + 4] =
        reinterpret_cast<unsigned long long>(slot.metadata);
  }
}

void append_ranges(const std::vector<Sequence>& sequences,
                   const int starts[BATCH],
                   const int counts[BATCH],
                   const __nv_bfloat16* ape,
                   unsigned long long* state_ptrs) {
  int* cu = managed<int>(BATCH + 1);
  cu[0] = 0;
  for (int sequence = 0; sequence < BATCH; ++sequence) {
    cu[sequence + 1] = cu[sequence] + counts[sequence];
  }
  const int total = cu[BATCH];
  auto* keys = managed<__nv_bfloat16>(total * DIM);
  auto* gates = managed<__nv_bfloat16>(total * DIM);
  int* positions = managed<int>(total);
  auto* valid = managed<unsigned char>(total);
  for (int sequence = 0; sequence < BATCH; ++sequence) {
    for (int local = 0; local < counts[sequence]; ++local) {
      const int source = starts[sequence] + local;
      const int target = cu[sequence] + local;
      positions[target] = sequences[sequence].positions[source];
      valid[target] = sequences[sequence].valid[source];
      std::copy_n(sequences[sequence].keys.data() + source * DIM, DIM,
                  keys + target * DIM);
      std::copy_n(sequences[sequence].gates.data() + source * DIM, DIM,
                  gates + target * DIM);
    }
  }
  glm53_dsa_pool_append<<<BATCH, DIM>>>(keys, gates, ape, cu, positions,
                                       valid, state_ptrs, BATCH);
  check(cudaGetLastError(), "DSA pool append launch");
  check(cudaDeviceSynchronize(), "DSA pool append synchronize");
  cudaFree(keys);
  cudaFree(gates);
  cudaFree(positions);
  cudaFree(valid);
  cudaFree(cu);
}

void reference_append(const Sequence& sequence,
                      const __nv_bfloat16* ape,
                      Slot* slot) {
  for (int token = 0; token < sequence.size(); ++token) {
    if (sequence.valid[token] == 0) continue;
    const int tail = slot->metadata[3];
    for (int channel = 0; channel < DIM; ++channel) {
      const int token_index = token * DIM + channel;
      if (tail < 3) {
        slot->tail_keys[tail * DIM + channel] = sequence.keys[token_index];
        slot->tail_gates[tail * DIM + channel] = sequence.gates[token_index];
      } else {
        float exponent[4];
        float maximum = -INFINITY;
        for (int lane = 0; lane < 4; ++lane) {
          const float gate = lane == 3
                                 ? __bfloat162float(sequence.gates[token_index])
                                 : __bfloat162float(
                                       slot->tail_gates[lane * DIM + channel]);
          exponent[lane] =
              gate + __bfloat162float(ape[lane * DIM + channel]);
          maximum = std::max(maximum, exponent[lane]);
        }
        float denominator = 0.0f;
        for (float& value : exponent) {
          value = std::exp(value - maximum);
          denominator += value;
        }
        float sum = 0.0f;
        for (int lane = 0; lane < 4; ++lane) {
          const float probability = bf16_round_host(exponent[lane] / denominator);
          const float key = lane == 3
                                ? __bfloat162float(sequence.keys[token_index])
                                : __bfloat162float(
                                      slot->tail_keys[lane * DIM + channel]);
          sum += bf16_round_host(probability * key);
        }
        slot->pooled[slot->metadata[2] * DIM + channel] =
            __float2bfloat16_rn(sum);
      }
    }
    if (slot->metadata[1] == 0) slot->metadata[0] = sequence.positions[token];
    ++slot->metadata[1];
    if (tail == 3) {
      ++slot->metadata[2];
      slot->metadata[3] = 0;
    } else {
      slot->metadata[3] = tail + 1;
    }
  }
}

float compare_slots(const Slot& left, const Slot& right) {
  float maximum = 0.0f;
  for (int index = 0; index < 4; ++index) {
    if (left.metadata[index] != right.metadata[index]) return INFINITY;
  }
  for (int index = 0; index < left.metadata[2] * DIM; ++index) {
    maximum = std::max(maximum,
                       std::fabs(__bfloat162float(left.pooled[index]) -
                                 __bfloat162float(right.pooled[index])));
  }
  for (int index = 0; index < left.metadata[3] * DIM; ++index) {
    maximum = std::max(maximum,
                       std::fabs(__bfloat162float(left.tail_keys[index]) -
                                 __bfloat162float(right.tail_keys[index])));
    maximum = std::max(maximum,
                       std::fabs(__bfloat162float(left.tail_gates[index]) -
                                 __bfloat162float(right.tail_gates[index])));
  }
  return maximum;
}

Sequence make_sequence(int sequence, int first_position, int tokens,
                       int left_padding) {
  Sequence result;
  result.keys.resize(tokens * DIM);
  result.gates.resize(tokens * DIM);
  result.positions.resize(tokens);
  result.valid.resize(tokens);
  for (int token = 0; token < tokens; ++token) {
    result.positions[token] = first_position + token;
    result.valid[token] = token >= left_padding;
    for (int channel = 0; channel < DIM; ++channel) {
      const int index = token * DIM + channel;
      result.keys[index] = __float2bfloat16_rn(
          sample(sequence * 100000 + token * DIM + channel, 0.9f));
      result.gates[index] = __float2bfloat16_rn(
          sample(sequence * 200000 + token * DIM + channel, 0.5f));
    }
  }
  return result;
}

}  // namespace

int main() {
  std::vector<Slot> slots(6);
  auto* state_ptrs = managed<unsigned long long>(BATCH * 5);
  auto* ape = managed<__nv_bfloat16>(4 * DIM);
  for (int index = 0; index < 4 * DIM; ++index) {
    ape[index] = __float2bfloat16_rn(sample(index + 7001, 0.25f));
  }
  const std::vector<Sequence> sequences = {
      make_sequence(0, 0, 11, 2), make_sequence(1, 5, 11, 0)};

  set_state_ptrs(state_ptrs, slots, 0, 2);
  const int starts_all[BATCH] = {0, 0};
  const int counts_all[BATCH] = {11, 11};
  append_ranges(sequences, starts_all, counts_all, ape, state_ptrs);

  set_state_ptrs(state_ptrs, slots, 3, 1);
  const int starts_a[BATCH] = {0, 0};
  const int counts_a[BATCH] = {4, 5};
  append_ranges(sequences, starts_a, counts_a, ape, state_ptrs);
  const int starts_b[BATCH] = {4, 5};
  const int counts_b[BATCH] = {3, 2};
  append_ranges(sequences, starts_b, counts_b, ape, state_ptrs);
  const int starts_c[BATCH] = {7, 7};
  const int counts_c[BATCH] = {4, 4};
  append_ranges(sequences, starts_c, counts_c, ape, state_ptrs);

  reference_append(sequences[0], ape, &slots[4]);
  reference_append(sequences[1], ape, &slots[5]);
  const float chunk_error = std::max(compare_slots(slots[0], slots[3]),
                                     compare_slots(slots[2], slots[1]));
  const float reference_error = std::max(compare_slots(slots[0], slots[4]),
                                         compare_slots(slots[2], slots[5]));

  for (int pool = 0; pool < MAX_POOLS; ++pool) {
    std::fill(slots[0].pooled + pool * DIM,
              slots[0].pooled + (pool + 1) * DIM,
              __float2bfloat16_rn(0.0f));
    slots[0].pooled[pool * DIM] = __float2bfloat16_rn(pool / 128);
    slots[0].pooled[pool * DIM + 1] = __float2bfloat16_rn(pool % 128);
  }
  slots[0].metadata[0] = 7;
  slots[0].metadata[1] = MAX_POOLS * 4 + 3;
  slots[0].metadata[2] = MAX_POOLS;
  slots[0].metadata[3] = 3;
  set_state_ptrs(state_ptrs, slots, 0, 2);

  auto* query = managed<__nv_bfloat16>(BATCH * HEADS * DIM);
  auto* weights = managed<__nv_bfloat16>(BATCH * HEADS);
  auto* query_valid = managed<unsigned char>(BATCH);
  float* scores = managed<float>(BATCH * MAX_POOLS);
  int* output = managed<int>(BATCH * OUTPUT_COUNT);
  query[0] = __float2bfloat16_rn(256.0f);
  query[1] = __float2bfloat16_rn(1.0f);
  weights[0] = __float2bfloat16_rn(1.0f);
  query_valid[0] = 1;
  query_valid[1] = 0;
  glm53_dsa_score<<<dim3(MAX_POOLS, BATCH), 128>>>(
      query, weights, query_valid, state_ptrs, scores, MAX_POOLS, BATCH);
  check(cudaGetLastError(), "DSA score launch");
  glm53_dsa_topk_expand_decode<<<BATCH, 256>>>(
      scores, query_valid, state_ptrs, output, MAX_POOLS, BATCH);
  check(cudaGetLastError(), "DSA top-k launch");
  check(cudaDeviceSynchronize(), "DSA index synchronize");

  bool topk_exact = true;
  for (int rank = 0; rank < 512; ++rank) {
    const int pool = MAX_POOLS - 1 - rank;
    for (int lane = 0; lane < 4; ++lane) {
      topk_exact &= output[rank * 4 + lane] == 7 + pool * 4 + lane;
    }
  }
  for (int lane = 0; lane < 3; ++lane) {
    topk_exact &= output[2048 + lane] == 7 + MAX_POOLS * 4 + lane;
  }
  for (int index = 0; index < OUTPUT_COUNT; ++index) {
    topk_exact &= output[OUTPUT_COUNT + index] == -1;
  }

  const int latent_capacity = MAX_POOLS * 4 + 16;
  for (int position = 0; position < latent_capacity; ++position) {
    for (int channel = 0; channel < PROBE_LATENT; ++channel) {
      slots[0].latent[position * PROBE_LATENT + channel] =
          __float2bfloat16_rn(sample(position * PROBE_LATENT + channel + 9001,
                                    0.12f));
    }
  }
  auto* absorbed_query =
      managed<__nv_bfloat16>(PROBE_MLA_HEADS * PROBE_LATENT);
  auto* sparse_output =
      managed<__nv_bfloat16>(PROBE_MLA_HEADS * PROBE_LATENT);
  std::vector<__nv_bfloat16> expected_sparse(PROBE_MLA_HEADS * PROBE_LATENT);
  for (int index = 0; index < PROBE_MLA_HEADS * PROBE_LATENT; ++index) {
    absorbed_query[index] = __float2bfloat16_rn(sample(index + 17001, 0.1f));
  }
  for (int head = 0; head < PROBE_MLA_HEADS; ++head) {
    std::vector<float> attention_scores;
    std::vector<int> positions;
    float maximum = -INFINITY;
    for (int rank = 0; rank < OUTPUT_COUNT; ++rank) {
      if (output[rank] < 0) continue;
      positions.push_back(output[rank]);
      float dot = 0.0f;
      for (int channel = 0; channel < PROBE_LATENT; ++channel) {
        dot += __bfloat162float(
                   absorbed_query[head * PROBE_LATENT + channel]) *
               __bfloat162float(
                   slots[0].latent[output[rank] * PROBE_LATENT + channel]);
      }
      attention_scores.push_back(dot * 0.0625f);
      maximum = std::max(maximum, attention_scores.back());
    }
    float denominator = 0.0f;
    for (float score : attention_scores) denominator += std::exp(score - maximum);
    for (int channel = 0; channel < PROBE_LATENT; ++channel) {
      float sum = 0.0f;
      for (size_t index = 0; index < positions.size(); ++index) {
        sum += std::exp(attention_scores[index] - maximum) *
               __bfloat162float(
                   slots[0].latent[positions[index] * PROBE_LATENT + channel]);
      }
      expected_sparse[head * PROBE_LATENT + channel] =
          __float2bfloat16_rn(sum / denominator);
    }
  }
  glm53_dsa_sparse_mla_decode<<<dim3(PROBE_MLA_HEADS, 1), 256>>>(
      absorbed_query, output, state_ptrs, sparse_output, PROBE_MLA_HEADS, 1,
      0.0625f);
  check(cudaGetLastError(), "DSA sparse MLA launch");
  check(cudaDeviceSynchronize(), "DSA sparse MLA synchronize");
  float sparse_max_abs = 0.0f;
  for (int index = 0; index < PROBE_MLA_HEADS * PROBE_LATENT; ++index) {
    sparse_max_abs = std::max(
        sparse_max_abs,
        std::fabs(__bfloat162float(sparse_output[index]) -
                  __bfloat162float(expected_sparse[index])));
  }

  std::printf(
      "pool_chunk_max_abs=%.9g pool_cpu_max_abs=%.9g "
      "left_pad_first=%d pools=%d tail=%d top512_exact=%s "
      "sparse_mla_max_abs=%.9g\n",
      chunk_error, reference_error, slots[0].metadata[0],
      slots[0].metadata[2], slots[0].metadata[3],
      topk_exact ? "true" : "false", sparse_max_abs);
  return chunk_error == 0.0f && reference_error <= 0.015625f && topk_exact &&
                 sparse_max_abs <= 0.002f
             ? 0
             : 1;
}
