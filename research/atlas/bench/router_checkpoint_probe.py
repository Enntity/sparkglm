#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Prove the FP32 GLM router against one real checkpoint gate."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = "model.language_model.layers.3.mlp.gate"
EXPERTS = 288
TOP_K = 8
HIDDEN = 4096


def tensor_file(model: Path, name: str) -> Path:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    return model / index["weight_map"][name]


def load(model: Path, name: str) -> torch.Tensor:
    with safe_open(tensor_file(model, name), framework="pt", device="cpu") as f:
        return f.get_tensor(name).cuda().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.rows <= 128:
        raise SystemExit("--rows must be in [1, 128]")

    gate = load(args.model, f"{ROOT}.weight")
    bias = load(args.model, f"{ROOT}.e_score_correction_bias")
    assert gate.dtype == torch.bfloat16 and tuple(gate.shape) == (EXPERTS, HIDDEN)
    assert bias.dtype == torch.float32 and tuple(bias.shape) == (EXPERTS,)

    torch.manual_seed(53)
    hidden = torch.randn(args.rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    logits = torch.empty(args.rows, EXPERTS, dtype=torch.float32, device="cuda")
    indices = torch.empty(args.rows, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.empty(args.rows, TOP_K, dtype=torch.float32, device="cuda")

    library = ctypes.CDLL(str(args.library))
    launch = library.atlas_glm53_router_launch
    launch.argtypes = [ctypes.c_uint64] * 6 + [ctypes.c_int, ctypes.c_uint64]
    launch.restype = ctypes.c_int
    status = launch(
        hidden.data_ptr(),
        gate.data_ptr(),
        bias.data_ptr(),
        logits.data_ptr(),
        indices.data_ptr(),
        weights.data_ptr(),
        args.rows,
        torch.cuda.current_stream().cuda_stream,
    )
    if status:
        raise RuntimeError(f"Atlas router launch failed with cudaError_t={status}")
    torch.cuda.synchronize()

    reference_logits = hidden.float() @ gate.float().t()
    scores = reference_logits.sigmoid()
    reference_indices = torch.topk(scores + bias, TOP_K, dim=-1).indices
    reference_weights = torch.gather(scores, 1, reference_indices)
    reference_weights = reference_weights / reference_weights.sum(dim=-1, keepdim=True) * 2.5
    bf16_indices = torch.topk(reference_logits.bfloat16().float().sigmoid() + bias, TOP_K, dim=-1).indices
    differing_bf16_rows = (bf16_indices != reference_indices).any(dim=-1).sum()

    logits_error = (logits - reference_logits).abs().max()
    weights_error = (weights - reference_weights).abs().max()
    ids_exact = torch.equal(indices.long(), reference_indices)
    result = {
        "rows": args.rows,
        "logits_max_abs": float(logits_error),
        "weights_max_abs": float(weights_error),
        "ids_exact": ids_exact,
        "rows_changed_by_bf16_logit_rounding": int(differing_bf16_rows),
    }
    print(result)
    if not ids_exact or logits_error > 0.002 or weights_error > 0.0001:
        raise SystemExit("GLM FP32 router parity gate failed")


if __name__ == "__main__":
    main()
