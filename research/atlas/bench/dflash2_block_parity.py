# SPDX-License-Identifier: AGPL-3.0-only

"""Localize Atlas DFlash2 numerical drift from a one-shot runtime dump.

The script loads only the small DFlash2 checkpoint. Atlas supplies the actual
query embeddings and context projections it consumed, so the 744 GB target
model is not needed for this comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_bf16(path: Path, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    words = np.fromfile(path, dtype=np.uint16).copy()
    expected = math.prod(shape)
    if words.size != expected:
        raise ValueError(f"{path}: got {words.size} BF16 values, expected {expected}")
    return torch.from_numpy(words).view(torch.bfloat16).reshape(shape).to(device)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(torch.bfloat16)


def grouped_conv(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    stage: int,
    groups: int,
) -> torch.Tensor:
    rows, width = hidden.shape
    blocks = hidden.float().reshape(rows, groups, width // groups)
    coefficients = (
        base[stage].float().reshape(1, 2, groups, width // groups)
        + dynamic.float().reshape(rows, 2, 2, groups)[:, stage, :, :, None]
    )
    out = coefficients[:, 0] * blocks
    out[1:] += coefficients[1:, 1] * blocks[:-1]
    return out.reshape(rows, width).to(torch.bfloat16)


def apply_neox_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[:, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[:, None, :]
    left, right = x.float().chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return (x.float() * cos + rotated * sin).to(torch.bfloat16)


def compare(label: str, reference: torch.Tensor, actual: torch.Tensor) -> None:
    ref = reference.float().flatten()
    got = actual.float().flatten()
    delta = got - ref
    cosine = F.cosine_similarity(ref, got, dim=0).item()
    rel_rms = delta.square().mean().sqrt().item() / max(
        ref.square().mean().sqrt().item(), 1e-12
    )
    print(
        f"{label:30s} cos={cosine:.8f} rel_rms={rel_rms:.6f} "
        f"max_abs={delta.abs().max().item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dump = args.dump_dir
    meta = json.loads((dump / "atlas_block_drafts.json").read_text())
    input_meta = json.loads((dump / "atlas_block_input_meta.json").read_text())
    gamma = int(meta["gamma"])
    hidden = int(meta["hidden_size"])
    vocab = int(meta["vocab_size"])
    position = int(meta["position"])
    layers = int(meta["num_drafter_layers"])
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    kv_dim = num_kv_heads * head_dim
    intermediate = 12288
    groups = hidden // 16
    eps = 1.0e-5
    theta = float(meta["rope_theta"])

    def dumped(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        return load_bf16(dump / name, shape, device)

    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        def weight(name: str) -> torch.Tensor:
            return checkpoint.get_tensor(name).to(device)

        print("context precompute")
        ctx_rows = (dump / "atlas_precompute_fc_proj.bin").stat().st_size // (2 * hidden)
        fc_proj = dumped("atlas_precompute_fc_proj.bin", (ctx_rows, hidden))
        fc_norm = rms_norm(fc_proj, weight("hidden_norm.weight"), eps)
        compare(
            "hidden_norm",
            fc_norm,
            dumped("atlas_precompute_fc_proj_normed.bin", (ctx_rows, hidden)),
        )

        fused_weight = torch.cat(
            [
                weight(f"layers.{layer}.self_attn.{kind}_proj.weight")
                for layer in range(layers)
                for kind in ("k", "v")
            ],
            dim=0,
        )
        fused_kv = F.linear(fc_norm, fused_weight)
        compare(
            "fused context K/V",
            fused_kv,
            dumped(
                "atlas_precompute_fused_kv_out.bin",
                (ctx_rows, layers * 2 * kv_dim),
            ),
        )
        fused_view = fused_kv.reshape(ctx_rows, layers, 2, num_kv_heads, head_dim)
        ctx_k0 = rms_norm(
            fused_view[:, 0, 0], weight("layers.0.self_attn.k_norm.weight"), eps
        )
        ctx_positions = torch.arange(ctx_rows, device=device)
        ctx_k0 = apply_neox_rope(ctx_k0, ctx_positions, theta)
        ctx_v0 = fused_view[:, 0, 1]
        compare(
            "context K0 post-RoPE",
            ctx_k0,
            dumped("atlas_precompute_layer0_k_post_rope.bin", (ctx_rows, num_kv_heads, head_dim)),
        )
        compare(
            "context V0",
            ctx_v0,
            dumped("atlas_precompute_layer0_v.bin", (ctx_rows, num_kv_heads, head_dim)),
        )

        print("layer 0 query block")
        stream = dumped("atlas_block_noise_embed.bin", (gamma, hidden))
        norm = rms_norm(stream, weight("layers.0.input_layernorm.weight"), eps)
        compare("input norm", norm, dumped("atlas_blk_L0_input_norm.bin", (gamma, hidden)))

        dynamic = F.linear(norm, weight("layers.0.attention_conv.kernel_projection.weight"))
        projection_input = grouped_conv(
            norm, dynamic, weight("layers.0.attention_conv.base_kernel"), 0, groups
        )
        q = F.linear(projection_input, weight("layers.0.self_attn.q_proj.weight"))
        k = F.linear(projection_input, weight("layers.0.self_attn.k_proj.weight"))
        v = F.linear(projection_input, weight("layers.0.self_attn.v_proj.weight"))
        compare("Q projection", q, dumped("atlas_blk_L0_q_postproj.bin", (gamma, hidden)))
        compare("K projection", k, dumped("atlas_blk_L0_k_postproj.bin", (gamma, kv_dim)))
        compare("V projection", v, dumped("atlas_blk_L0_v.bin", (gamma, kv_dim)))

        q = rms_norm(
            q.reshape(gamma, num_heads, head_dim),
            weight("layers.0.self_attn.q_norm.weight"),
            eps,
        )
        k = rms_norm(
            k.reshape(gamma, num_kv_heads, head_dim),
            weight("layers.0.self_attn.k_norm.weight"),
            eps,
        )
        compare("Q norm", q, dumped("atlas_blk_L0_q_postnorm.bin", q.shape))
        compare("K norm", k, dumped("atlas_blk_L0_k_postnorm.bin", k.shape))

        positions = torch.arange(position, position + gamma, device=device)
        q = apply_neox_rope(q, positions, theta)
        k = apply_neox_rope(k, positions, theta)
        compare("Q post-RoPE", q, dumped("atlas_blk_L0_q_postrope.bin", q.shape))
        compare("K post-RoPE", k, dumped("atlas_blk_L0_k_postrope.bin", k.shape))

        all_k = torch.cat((ctx_k0, k), dim=0).repeat_interleave(num_heads // num_kv_heads, dim=1)
        all_v = torch.cat((ctx_v0, v.reshape(gamma, num_kv_heads, head_dim)), dim=0)
        all_v = all_v.repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), all_k.float()) / math.sqrt(head_dim)
        attention = torch.einsum(
            "hqk,khd->qhd", scores.softmax(dim=-1), all_v.float()
        ).to(torch.bfloat16)
        compare(
            "attention output",
            attention,
            dumped("atlas_blk_L0_attn_out.bin", attention.shape),
        )

        attention_flat = attention.reshape(gamma, hidden)
        attention_out = F.linear(attention_flat, weight("layers.0.self_attn.o_proj.weight"))
        attention_out = grouped_conv(
            attention_out,
            dynamic,
            weight("layers.0.attention_conv.base_kernel"),
            1,
            groups,
        )
        residual = (stream.float() + attention_out.float()).to(torch.bfloat16)
        mlp_norm = rms_norm(residual, weight("layers.0.post_attention_layernorm.weight"), eps)
        mlp_dynamic = F.linear(mlp_norm, weight("layers.0.mlp_conv.kernel_projection.weight"))
        mlp_input = grouped_conv(
            mlp_norm, mlp_dynamic, weight("layers.0.mlp_conv.base_kernel"), 0, groups
        )
        gate = F.linear(mlp_input, weight("layers.0.mlp.gate_proj.weight"))
        up = F.linear(mlp_input, weight("layers.0.mlp.up_proj.weight"))
        mlp = F.linear(F.silu(gate) * up, weight("layers.0.mlp.down_proj.weight"))
        mlp = grouped_conv(
            mlp, mlp_dynamic, weight("layers.0.mlp_conv.base_kernel"), 1, groups
        )
        layer_out = (residual.float() + mlp.float()).to(torch.bfloat16)
        compare(
            "layer 0 output",
            layer_out,
            dumped("atlas_blk_L0_layer_out.bin", (gamma, hidden)),
        )

        next_norm = rms_norm(layer_out, weight("layers.1.input_layernorm.weight"), eps)
        compare(
            "layer 1 input norm",
            next_norm,
            dumped("atlas_blk_L1_input_norm.bin", (gamma, hidden)),
        )
        print(
            f"meta: ctx_rows={ctx_rows} position={position} gamma={gamma} "
            f"q_offset={input_meta['option_b_q_offset']} vocab={vocab}"
        )


if __name__ == "__main__":
    main()
