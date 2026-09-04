// SPDX-License-Identifier: MIT AND Apache-2.0
#pragma once

#include <torch/extension.h>

void exl3_fat_gemm(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_m64(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_pair(
    at::Tensor a,
    at::Tensor packed_gate,
    at::Tensor packed_up,
    at::Tensor out,
    at::Tensor svh_gate,
    at::Tensor svh_up,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_pair_m64(
    at::Tensor a,
    at::Tensor packed_gate,
    at::Tensor packed_up,
    at::Tensor out,
    at::Tensor svh_gate,
    at::Tensor svh_up,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_swiglu_had(
    at::Tensor gate_up,
    at::Tensor out,
    at::Tensor down_suh,
    double limit);

void exl3_fat_gemm_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_scatter_m64(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_grouped_prefill_k4(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_offsets,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& had_input,
    const at::Tensor& gate_up,
    const at::Tensor& had_down,
    const at::Tensor& tasks,
    const at::Tensor& task_count,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    int64_t cap,
    double act_limit);
