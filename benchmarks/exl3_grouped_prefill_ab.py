#!/usr/bin/env python3
"""Exact-shape A/B for the GPU-resident GLM-5.3 fat-expert prefill path.

This is intentionally model-free: it constructs valid synthetic K4 MCG
trellises at the production TP2 shard dimensions and compares the existing
per-expert launch sequence with exl3_grouped_prefill_k4.
"""
from __future__ import annotations

import argparse
import statistics

import torch

import exllamav3_ext as ext


HIDDEN = 4096
INTERMEDIATE = 1024
PACKED_WORDS = 64


def make_matrix(size_k: int, size_n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    trellis = torch.randint(
        -(1 << 15),
        1 << 15,
        (size_k // 16, size_n // 16, PACKED_WORDS),
        dtype=torch.int16,
        device="cuda",
        generator=generator,
    )
    suh = torch.empty(size_k, dtype=torch.float16, device="cuda").uniform_(
        0.004, 0.012, generator=generator
    )
    svh = torch.empty(size_n, dtype=torch.float16, device="cuda").uniform_(
        0.004, 0.012, generator=generator
    )
    return trellis, suh, svh


def ptrs(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([item.data_ptr() for item in items], dtype=torch.int64, device="cuda")


def event_ms(fn, iterations: int) -> list[float]:
    values: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--cap", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not (args.cap < args.rows <= args.tokens):
        raise SystemExit("require cap < rows <= tokens")

    torch.manual_seed(20260903)
    x = (torch.randn(args.tokens, HIDDEN, device="cuda") * 0.05).half()
    counts = torch.full((args.experts,), args.rows, dtype=torch.int64, device="cuda")
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device="cuda"), counts.cumsum(0))
    )
    token_parts = [
        (torch.arange(args.rows, device="cuda", dtype=torch.int64) + 17 * e)
        % args.tokens
        for e in range(args.experts)
    ]
    token_sorted = torch.cat(token_parts)
    weight_sorted = torch.full(
        (token_sorted.numel(),),
        1.0 / args.experts,
        dtype=torch.float16,
        device="cuda",
    )

    gate_t: list[torch.Tensor] = []
    gate_u: list[torch.Tensor] = []
    gate_v: list[torch.Tensor] = []
    up_t: list[torch.Tensor] = []
    up_v: list[torch.Tensor] = []
    down_t: list[torch.Tensor] = []
    down_u: list[torch.Tensor] = []
    down_v: list[torch.Tensor] = []
    for e in range(args.experts):
        gt, gu, gv = make_matrix(HIDDEN, INTERMEDIATE, 1000 + e)
        ut, _, uv = make_matrix(HIDDEN, INTERMEDIATE, 2000 + e)
        dt, du, dv = make_matrix(INTERMEDIATE, HIDDEN, 3000 + e)
        gate_t.append(gt)
        gate_u.append(gu)
        gate_v.append(gv)
        up_t.append(ut)
        up_v.append(uv)
        down_t.append(dt)
        down_u.append(du)
        down_v.append(dv)

    h = torch.empty(args.rows, HIDDEN, dtype=torch.float16, device="cuda")
    h13 = torch.empty_like(h)
    gate_up = torch.empty(
        args.rows, 2 * INTERMEDIATE, dtype=torch.float32, device="cuda"
    )
    h2 = torch.empty(args.rows, INTERMEDIATE, dtype=torch.float16, device="cuda")
    grouped_h13 = torch.empty(
        token_sorted.numel(), HIDDEN, dtype=torch.float16, device="cuda"
    )
    grouped_gate_up = torch.empty(
        token_sorted.numel(), 2 * INTERMEDIATE, dtype=torch.float32, device="cuda"
    )
    grouped_h2 = torch.empty(
        token_sorted.numel(), INTERMEDIATE, dtype=torch.float16, device="cuda"
    )
    task_capacity = (token_sorted.numel() + 63) // 64 + args.experts
    tasks = torch.empty(task_capacity, 4, dtype=torch.int32, device="cuda")
    task_count = torch.empty(1, dtype=torch.int32, device="cuda")
    baseline_out = torch.empty(args.tokens, HIDDEN, dtype=torch.float32, device="cuda")
    grouped_out = torch.empty_like(baseline_out)

    tables = (
        ptrs(gate_t),
        ptrs(gate_u),
        ptrs(gate_v),
        ptrs(up_t),
        ptrs(up_v),
        ptrs(down_t),
        ptrs(down_u),
        ptrs(down_v),
    )

    def baseline() -> None:
        baseline_out.zero_()
        for expert in range(args.experts):
            start = expert * args.rows
            stop = start + args.rows
            torch.index_select(x, 0, token_sorted[start:stop], out=h)
            ext.had_r_128(h, h13, gate_u[expert], None, 1.0)
            ext.exl3_fat_gemm_pair_m64(
                h13,
                gate_t[expert],
                up_t[expert],
                gate_up,
                gate_v[expert],
                up_v[expert],
                4,
                True,
                False,
            )
            ext.exl3_fat_swiglu_had(gate_up, h2, down_u[expert], 10.0)
            ext.exl3_fat_gemm_scatter_m64(
                h2,
                down_t[expert],
                baseline_out,
                down_v[expert],
                token_sorted[start:stop],
                weight_sorted[start:stop],
                4,
                True,
                False,
            )

    def grouped() -> None:
        grouped_out.zero_()
        ext.exl3_grouped_prefill_k4(
            x,
            grouped_out,
            offsets,
            token_sorted,
            weight_sorted,
            grouped_h13,
            grouped_gate_up,
            grouped_h2,
            tasks,
            task_count,
            *tables,
            args.cap,
            10.0,
        )

    baseline()
    grouped()
    torch.cuda.synchronize()
    difference = (baseline_out - grouped_out).abs()
    print(
        "correctness",
        f"max_abs={difference.max().item():.9g}",
        f"mean_abs={difference.mean().item():.9g}",
        f"exact={torch.equal(baseline_out, grouped_out)}",
    )
    torch.testing.assert_close(grouped_out, baseline_out, rtol=1e-5, atol=1e-6)

    for _ in range(args.warmup):
        baseline()
        grouped()
    torch.cuda.synchronize()
    baseline_ms = event_ms(baseline, args.iterations)
    grouped_ms = event_ms(grouped, args.iterations)
    base_med = statistics.median(baseline_ms)
    group_med = statistics.median(grouped_ms)
    print(
        "timing",
        f"baseline_median_ms={base_med:.3f}",
        f"grouped_median_ms={group_med:.3f}",
        f"speedup={base_med / group_med:.3f}x",
        f"improvement={(base_med / group_med - 1.0) * 100.0:.1f}%",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
