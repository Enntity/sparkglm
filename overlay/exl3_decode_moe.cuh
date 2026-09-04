// SPDX-License-Identifier: MIT AND Apache-2.0
#pragma once

#include <torch/extension.h>

// K4-only, TP2-local GLM-5.3 decode experiment. The output tensor is written
// in-place so the Python caller can retain its existing FP32 MoE contract.
void exl3_decode_moe_k4(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_offsets,
    const at::Tensor& expert_sorted,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& had_gate,
    const at::Tensor& had_up,
    const at::Tensor& gate,
    const at::Tensor& up,
    const at::Tensor& had_down,
    const at::Tensor& down,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    double act_limit);
