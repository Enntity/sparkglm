// SPDX-License-Identifier: AGPL-3.0-only

// Checkpoint-free GB10 A/B for GLM-5.3 mHC pre. The reference is Atlas's
// production scalar BF16-weight/FP32-highway kernel. The candidate computes
// the 24 learned mix logits as one TF32 tensor-core GEMM, with the RMS sum and
// Sinkhorn/collapse kept in small CUDA kernels.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HC_FN_TYPE __nv_bfloat16
#include "../../kernels/gb10/glm-5.3-flash/exl3/hyper_connection.cu"

namespace {

constexpr unsigned int kHidden = 4096;
constexpr unsigned int kHc = 4;
constexpr unsigned int kMix = (2 + kHc) * kHc;
constexpr unsigned int kWidth = kHidden * kHc;
constexpr unsigned int kBlock = 256;

#define CUDA_OK(call)                                                         \
  do {                                                                        \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                \
                   cudaGetErrorString(status));                                \
      std::exit(1);                                                            \
    }                                                                         \
  } while (0)

#define CUBLAS_OK(call)                                                       \
  do {                                                                        \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "%s:%d: cuBLAS status %d\n", __FILE__, __LINE__,   \
                   static_cast<int>(status));                                  \
      std::exit(1);                                                            \
    }                                                                         \
  } while (0)

float elapsed_us(cudaEvent_t begin, cudaEvent_t end, int iterations) {
  CUDA_OK(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CUDA_OK(cudaEventElapsedTime(&milliseconds, begin, end));
  return milliseconds * 1000.0f / static_cast<float>(iterations);
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 8;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 200;
  if (rows < 1 || rows > 64) return 2;

  std::vector<float> residual_host(static_cast<size_t>(rows) * kWidth);
  std::vector<float> fn_float_host(static_cast<size_t>(kMix) * kWidth);
  std::vector<__nv_bfloat16> fn_bf16_host(fn_float_host.size());
  std::vector<float> scale_host{0.125f, 0.125f, 0.125f};
  std::vector<float> base_host(kMix);
  for (size_t i = 0; i < residual_host.size(); ++i) {
    residual_host[i] =
        0.20f * std::sin(static_cast<float>(i) * 0.017123f) +
        0.03f * std::cos(static_cast<float>(i) * 0.031337f);
  }
  for (size_t i = 0; i < fn_float_host.size(); ++i) {
    const float value =
        0.025f * std::sin(static_cast<float>(i) * 0.013579f) +
        0.004f * std::cos(static_cast<float>(i) * 0.029771f);
    fn_bf16_host[i] = __float2bfloat16(value);
    fn_float_host[i] = __bfloat162float(fn_bf16_host[i]);
  }
  for (size_t i = 0; i < base_host.size(); ++i) {
    base_host[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 32.0f;
  }

  float *residual, *fn_float, *scale, *base, *mix, *sqsum, *post_ref, *comb_ref,
      *post_candidate, *comb_candidate;
  __nv_bfloat16 *fn_bf16, *output_ref, *output_candidate;
  CUDA_OK(cudaMalloc(&residual, residual_host.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&fn_float, fn_float_host.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&fn_bf16, fn_bf16_host.size() * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMalloc(&scale, 3 * sizeof(float)));
  CUDA_OK(cudaMalloc(&base, kMix * sizeof(float)));
  CUDA_OK(cudaMalloc(&mix, static_cast<size_t>(rows) * kMix * sizeof(float)));
  CUDA_OK(cudaMalloc(&sqsum, rows * sizeof(float)));
  CUDA_OK(cudaMalloc(&post_ref, rows * kHc * sizeof(float)));
  CUDA_OK(cudaMalloc(&comb_ref, rows * kHc * kHc * sizeof(float)));
  CUDA_OK(cudaMalloc(&post_candidate, rows * kHc * sizeof(float)));
  CUDA_OK(cudaMalloc(&comb_candidate, rows * kHc * kHc * sizeof(float)));
  CUDA_OK(cudaMalloc(&output_ref, static_cast<size_t>(rows) * kHidden * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMalloc(&output_candidate,
                     static_cast<size_t>(rows) * kHidden * sizeof(__nv_bfloat16)));
  CUDA_OK(cudaMemcpy(residual, residual_host.data(), residual_host.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(fn_float, fn_float_host.data(), fn_float_host.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(fn_bf16, fn_bf16_host.data(),
                     fn_bf16_host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(scale, scale_host.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(base, base_host.data(), kMix * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_OK(cublasCreate(&handle));
  CUBLAS_OK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto candidate = [&]() {
    glm53_hc_pre_sqsum<<<rows, kBlock>>>(residual, sqsum);
    CUBLAS_OK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, kMix, rows, kWidth, &alpha, fn_float,
        CUDA_R_32F, kWidth, residual, CUDA_R_32F, kWidth, &beta, mix, CUDA_R_32F,
        kMix, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    glm53_hc_pre_finish<<<rows, kBlock>>>(
        residual, mix, sqsum, scale, base, output_candidate, post_candidate,
        comb_candidate);
  };
  for (int i = 0; i < 20; ++i) {
    hc_pre<<<rows, kBlock>>>(residual, fn_bf16, scale, base, output_ref, post_ref,
                            comb_ref, kHidden, kHc, 20, 1.0e-5f, 1.0e-6f);
    candidate();
  }
  CUDA_OK(cudaDeviceSynchronize());

  cudaEvent_t begin, end;
  CUDA_OK(cudaEventCreate(&begin));
  CUDA_OK(cudaEventCreate(&end));
  CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) {
    hc_pre<<<rows, kBlock>>>(residual, fn_bf16, scale, base, output_ref, post_ref,
                            comb_ref, kHidden, kHc, 20, 1.0e-5f, 1.0e-6f);
  }
  CUDA_OK(cudaEventRecord(end));
  const float reference_us = elapsed_us(begin, end, iterations);
  CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) candidate();
  CUDA_OK(cudaEventRecord(end));
  const float candidate_us = elapsed_us(begin, end, iterations);

  std::vector<__nv_bfloat16> ref_host(static_cast<size_t>(rows) * kHidden);
  std::vector<__nv_bfloat16> candidate_host(ref_host.size());
  std::vector<float> post_ref_host(static_cast<size_t>(rows) * kHc);
  std::vector<float> post_candidate_host(post_ref_host.size());
  std::vector<float> comb_ref_host(static_cast<size_t>(rows) * kHc * kHc);
  std::vector<float> comb_candidate_host(comb_ref_host.size());
  CUDA_OK(cudaMemcpy(ref_host.data(), output_ref,
                     ref_host.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(candidate_host.data(), output_candidate,
                     candidate_host.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(post_ref_host.data(), post_ref,
                     post_ref_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(post_candidate_host.data(), post_candidate,
                     post_candidate_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(comb_ref_host.data(), comb_ref,
                     comb_ref_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(comb_candidate_host.data(), comb_candidate,
                     comb_candidate_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  float max_abs = 0.0f;
  double mean_abs = 0.0;
  for (size_t i = 0; i < ref_host.size(); ++i) {
    const float difference =
        std::abs(__bfloat162float(ref_host[i]) - __bfloat162float(candidate_host[i]));
    max_abs = std::max(max_abs, difference);
    mean_abs += difference;
  }
  mean_abs /= static_cast<double>(ref_host.size());
  float post_max_abs = 0.0f;
  double post_mean_abs = 0.0;
  for (size_t i = 0; i < post_ref_host.size(); ++i) {
    const float difference = std::abs(post_ref_host[i] - post_candidate_host[i]);
    post_max_abs = std::max(post_max_abs, difference);
    post_mean_abs += difference;
  }
  post_mean_abs /= static_cast<double>(post_ref_host.size());
  float comb_max_abs = 0.0f;
  double comb_mean_abs = 0.0;
  for (size_t i = 0; i < comb_ref_host.size(); ++i) {
    const float difference = std::abs(comb_ref_host[i] - comb_candidate_host[i]);
    comb_max_abs = std::max(comb_max_abs, difference);
    comb_mean_abs += difference;
  }
  comb_mean_abs /= static_cast<double>(comb_ref_host.size());
  std::printf(
      "rows=%d scalar_us=%.3f tf32_pipeline_us=%.3f speedup=%.3f "
      "output_max_abs=%.7f output_mean_abs=%.9f "
      "post_max_abs=%.9f post_mean_abs=%.11f "
      "comb_max_abs=%.9f comb_mean_abs=%.11f\n",
      rows, reference_us, candidate_us, reference_us / candidate_us, max_abs,
      mean_abs, post_max_abs, post_mean_abs, comb_max_abs, comb_mean_abs);
  return 0;
}
