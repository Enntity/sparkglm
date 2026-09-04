#!/usr/bin/env python3
"""Measure the crossover for JIT-promoting one GLM EXL3 expert.

The current direct K4 kernel decodes a packed weight tile for every M64 row
tile.  This model-free gate compares it with reconstructing the expert once
and using the dense tensor-core GEMM.  It deliberately reports three paths:

* direct_m64: the production paired M64 gate/up kernel;
* direct_m128: the existing paired M128 kernel, to test whether taller tiles
  amortize trellis decode better for very hot experts;
* promote: reconstruct + dense GEMM, excluding an avoidable packed concat;
* promote_copy: the same path including today's two-tensor concat cost;
* cached: dense GEMM with an already promoted expert, an upper-bound for a
  small cross-chunk expert cache.

The production implementation should use a pointer-aware paired reconstruct,
so ``promote`` is the relevant first feasibility gate.  ``promote_copy`` is a
conservative implementation bound, while ``cached`` prices the more radical
reuse option.
"""
from __future__ import annotations

import argparse
import statistics

import torch

import exllamav3_ext as ext
import vllm._custom_ops  # noqa: F401 - registers fused FP8 quantization ops


HIDDEN = 4096
INTERMEDIATE = 1024
PACKED_WORDS = 64


def make_matrix(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    trellis = torch.randint(
        -(1 << 15),
        1 << 15,
        (HIDDEN // 16, INTERMEDIATE // 16, PACKED_WORDS),
        dtype=torch.int16,
        device="cuda",
        generator=generator,
    )
    svh = torch.empty(
        INTERMEDIATE, dtype=torch.float16, device="cuda"
    ).uniform_(0.004, 0.012, generator=generator)
    return trellis, svh


def event_ms(fn, iterations: int) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", default="64,128,256,384,512,768,1024,1536,2048"
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    rows = [int(value) for value in args.rows.split(",")]
    if not rows or min(rows) < 1:
        raise SystemExit("--rows must contain positive integers")

    torch.manual_seed(20260903)
    gate_t, gate_v = make_matrix(1000)
    up_t, up_v = make_matrix(2000)
    packed13 = torch.empty(
        HIDDEN // 16,
        2 * INTERMEDIATE // 16,
        PACKED_WORDS,
        dtype=torch.int16,
        device="cuda",
    )
    packed13[:, : INTERMEDIATE // 16].copy_(gate_t)
    packed13[:, INTERMEDIATE // 16 :].copy_(up_t)
    svh13 = torch.cat((gate_v, up_v)).contiguous()
    w13 = torch.empty(
        HIDDEN, 2 * INTERMEDIATE, dtype=torch.float16, device="cuda"
    )
    ext.reconstruct(w13, packed13, 4, True, False)
    torch.cuda.synchronize()

    print(
        "rows direct_m64_ms direct_m128_ms promote_ms promote_copy_ms cached_ms "
        "fp8_cached_w_ms fp8_floor_ms m64_over_m128 direct_over_promote direct_over_copy "
        "direct_over_cached direct_over_fp8_cached_w direct_over_fp8_floor "
        "max_abs cosine fp8_max_abs fp8_cosine",
        flush=True,
    )
    for row_count in rows:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(3000 + row_count)
        x = torch.empty(
            row_count, HIDDEN, dtype=torch.float16, device="cuda"
        ).normal_(0.0, 0.05, generator=generator)
        direct_out = torch.empty(
            row_count, 2 * INTERMEDIATE, dtype=torch.float32, device="cuda"
        )
        promote_out = torch.empty_like(direct_out)

        def direct() -> None:
            ext.exl3_fat_gemm_pair_m64(
                x,
                gate_t,
                up_t,
                direct_out,
                gate_v,
                up_v,
                4,
                True,
                False,
            )

        def direct_m128() -> None:
            ext.exl3_fat_gemm_pair(
                x,
                gate_t,
                up_t,
                direct_out,
                gate_v,
                up_v,
                4,
                True,
                False,
            )

        def dense() -> None:
            ext.hgemm(x, w13, promote_out)
            ext.had_r_128(promote_out, promote_out, None, svh13, 1.0)

        def promote() -> None:
            ext.reconstruct(w13, packed13, 4, True, False)
            dense()

        def promote_copy() -> None:
            packed13[:, : INTERMEDIATE // 16].copy_(gate_t)
            packed13[:, INTERMEDIATE // 16 :].copy_(up_t)
            promote()

        # Price the best plausible FP8 variant before writing a direct
        # trellis-to-FP8 reconstruction kernel.  The weight and its scale are
        # treated as cached; fp8_cached_w still quantizes the per-call input,
        # while fp8_floor also caches the input and is an intentionally
        # impossible lower bound for this isolated expert.
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        weight_scale = w13.abs().max().float() / fp8_max
        weight_fp8 = (w13 / weight_scale).clamp(-fp8_max, fp8_max).to(
            torch.float8_e4m3fn
        )
        input_scale = torch.empty(1, dtype=torch.float32, device="cuda")
        input_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        torch.ops._C.dynamic_scaled_fp8_quant(input_fp8, x, input_scale)
        fp8_out = torch.empty_like(direct_out)

        def fp8_dense(input_quantized: torch.Tensor) -> None:
            torch._scaled_mm(
                input_quantized,
                weight_fp8,
                input_scale,
                weight_scale,
                out_dtype=torch.float32,
                out=fp8_out,
            )
            ext.had_r_128(fp8_out, fp8_out, None, svh13, 1.0)

        def fp8_cached_weight() -> None:
            torch.ops._C.dynamic_scaled_fp8_quant(input_fp8, x, input_scale)
            fp8_dense(input_fp8)

        def fp8_floor() -> None:
            fp8_dense(input_fp8)

        direct()
        promote()
        torch.cuda.synchronize()
        difference = (direct_out - promote_out).abs()
        max_abs = float(difference.max().item())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                direct_out.float().flatten(),
                promote_out.float().flatten(),
                dim=0,
            ).item()
        )
        fp8_floor()
        torch.cuda.synchronize()
        fp8_difference = (direct_out - fp8_out).abs()
        fp8_max_abs = float(fp8_difference.max().item())
        fp8_cosine = float(
            torch.nn.functional.cosine_similarity(
                direct_out.float().flatten(), fp8_out.float().flatten(), dim=0
            ).item()
        )

        for _ in range(args.warmup):
            direct()
            direct_m128()
            promote()
            promote_copy()
            dense()
            fp8_cached_weight()
            fp8_floor()
        torch.cuda.synchronize()

        # Reverse the order in the second half to reduce thermal/order bias.
        direct_samples = event_ms(direct, args.iterations // 2)
        direct_m128_samples = event_ms(direct_m128, args.iterations // 2)
        promote_samples = event_ms(promote, args.iterations // 2)
        copy_samples = event_ms(promote_copy, args.iterations // 2)
        cached_samples = event_ms(dense, args.iterations // 2)
        fp8_cached_weight_samples = event_ms(
            fp8_cached_weight, args.iterations // 2
        )
        fp8_floor_samples = event_ms(fp8_floor, args.iterations // 2)
        fp8_floor_samples += event_ms(
            fp8_floor, args.iterations - args.iterations // 2
        )
        fp8_cached_weight_samples += event_ms(
            fp8_cached_weight, args.iterations - args.iterations // 2
        )
        cached_samples += event_ms(dense, args.iterations - args.iterations // 2)
        copy_samples += event_ms(
            promote_copy, args.iterations - args.iterations // 2
        )
        promote_samples += event_ms(promote, args.iterations - args.iterations // 2)
        direct_m128_samples += event_ms(
            direct_m128, args.iterations - args.iterations // 2
        )
        direct_samples += event_ms(direct, args.iterations - args.iterations // 2)

        direct_med = statistics.median(direct_samples)
        direct_m128_med = statistics.median(direct_m128_samples)
        promote_med = statistics.median(promote_samples)
        copy_med = statistics.median(copy_samples)
        cached_med = statistics.median(cached_samples)
        fp8_cached_weight_med = statistics.median(fp8_cached_weight_samples)
        fp8_floor_med = statistics.median(fp8_floor_samples)
        print(
            row_count,
            f"{direct_med:.4f}",
            f"{direct_m128_med:.4f}",
            f"{promote_med:.4f}",
            f"{copy_med:.4f}",
            f"{cached_med:.4f}",
            f"{fp8_cached_weight_med:.4f}",
            f"{fp8_floor_med:.4f}",
            f"{direct_med / direct_m128_med:.3f}",
            f"{direct_med / promote_med:.3f}",
            f"{direct_med / copy_med:.3f}",
            f"{direct_med / cached_med:.3f}",
            f"{direct_med / fp8_cached_weight_med:.3f}",
            f"{direct_med / fp8_floor_med:.3f}",
            f"{max_abs:.7g}",
            f"{cosine:.9f}",
            f"{fp8_max_abs:.7g}",
            f"{fp8_cosine:.9f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
