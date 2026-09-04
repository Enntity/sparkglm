#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Check Atlas's BF16 DSA projection orientation against one real tensor.

Loads one 32 MiB kv_b projection, not the full GLM checkpoint.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch
from safetensors import safe_open


TENSOR = "model.language_model.layers.3.self_attn.kv_b_proj.weight"
HEADS = 64
QK_DIM = 256
V_DIM = 256
LATENT = 512


def tensor_file(model: Path, name: str) -> Path:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    return model / index["weight_map"][name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 8:
        raise SystemExit("--tokens must be in [1, 8]")

    with safe_open(tensor_file(args.model, TENSOR), framework="pt", device="cpu") as f:
        weight = f.get_tensor(TENSOR).cuda().contiguous()
    assert weight.dtype == torch.bfloat16
    assert tuple(weight.shape) == (HEADS * (QK_DIM + V_DIM), LATENT)
    weight_h = weight.view(HEADS, QK_DIM + V_DIM, LATENT)

    torch.manual_seed(53)
    query = (torch.randn(args.tokens, HEADS, QK_DIM, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    latent = (torch.randn(args.tokens, HEADS, LATENT, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    absorbed = torch.empty(
        args.tokens, HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    expanded = torch.empty(
        args.tokens, HEADS, V_DIM, dtype=torch.bfloat16, device="cuda"
    )

    library = ctypes.CDLL(str(args.library))
    launch = library.atlas_glm53_dsa_projection_launch
    launch.argtypes = [ctypes.c_uint64] * 5 + [ctypes.c_int, ctypes.c_uint64]
    launch.restype = ctypes.c_int
    stream = torch.cuda.current_stream().cuda_stream
    status = launch(
        query.data_ptr(),
        latent.data_ptr(),
        weight.data_ptr(),
        absorbed.data_ptr(),
        expanded.data_ptr(),
        args.tokens,
        stream,
    )
    if status:
        raise RuntimeError(f"Atlas DSA projection launch failed with cudaError_t={status}")
    torch.cuda.synchronize()

    # FP32 inputs and weights make the oracle independent of BF16 GEMM modes.
    absorb_ref = torch.einsum(
        "thq,hql->thl", query.float(), weight_h[:, :QK_DIM, :].float()
    ).to(torch.bfloat16)
    expand_ref = torch.einsum(
        "thl,hvl->thv", latent.float(), weight_h[:, QK_DIM:, :].float()
    ).to(torch.bfloat16)
    absorb_delta = (absorbed.float() - absorb_ref.float()).abs()
    expand_delta = (expanded.float() - expand_ref.float()).abs()
    result = {
        "tokens": args.tokens,
        "weight_mib": round(weight.nbytes / 2**20, 2),
        "absorb_max_abs": float(absorb_delta.max()),
        "expand_max_abs": float(expand_delta.max()),
        "finite": bool(torch.isfinite(absorbed).all() and torch.isfinite(expanded).all()),
    }
    print(result)
    if not result["finite"] or max(result["absorb_max_abs"], result["expand_max_abs"]) > 0.02:
        raise SystemExit("DSA BF16 projection parity gate failed")


if __name__ == "__main__":
    main()
