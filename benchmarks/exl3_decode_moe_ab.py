#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Numerical and latency A/B for the exact GLM-5.3 TP2 EXL3 decode shape."""

from __future__ import annotations

import argparse
import os
import statistics
import types

import torch


def make_layer(device: torch.device, max_tokens: int = 16, experts: int = 16):
    from vllm.model_executor.layers.quantization.exl3 import (
        Exl3Config,
        Exl3MoEMethod,
    )

    os.environ["SPARKGLM_TINY_DUMMY"] = "1"
    os.environ["EXL3_DECODE_COOP_K4"] = "1"
    os.environ["EXL3_DECODE_COOP_MAX_TOKENS"] = str(max_tokens)
    moe = types.SimpleNamespace(swiglu_limit=75.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=experts,
        hidden_size=4096,
        intermediate_size_per_partition=1024,
        params_dtype=torch.float16,
    )
    layer = layer.to(device)
    method.process_weights_after_loading(layer)
    if layer._exl3_decode_coop_scratch is None:
        raise RuntimeError("cooperative K4 scratch was not created")
    return layer


def one_call(layer, x, ids, weights, *, candidate: bool):
    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_experts

    scratch = layer._exl3_decode_coop_scratch
    if not candidate:
        layer._exl3_decode_coop_scratch = None
    try:
        return apply_exl3_experts(x, ids, weights, layer, fused=True)
    finally:
        layer._exl3_decode_coop_scratch = scratch


def time_ms(layer, x, ids, weights, *, candidate: bool, repeats: int) -> list[float]:
    values = []
    for _ in range(5):
        one_call(layer, x, ids, weights, candidate=candidate)
    torch.cuda.synchronize()
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        one_call(layer, x, ids, weights, candidate=candidate)
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, choices=(16,32), default=16)
    parser.add_argument("--experts", type=int, choices=(16,288), default=16)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    layer = make_layer(device, args.max_tokens, args.experts)
    failed = False
    print("tokens baseline_ms candidate_ms speedup max_abs mean_abs cosine", flush=True)
    for tokens in (1, 2, 4, 8, 16, 17, 31, 32, 33):
        x = torch.randn(tokens, 4096, dtype=torch.float16, device=device)
        row = torch.arange(tokens, device=device)[:, None]
        slot = torch.arange(8, device=device)[None, :]
        ids = ((row * 5 + slot * 3) % args.experts).to(torch.long)
        raw = torch.rand(tokens, 8, dtype=torch.float32, device=device)
        weights = torch.softmax(raw, dim=-1).half()
        baseline = one_call(layer, x, ids, weights, candidate=False)
        candidate = one_call(layer, x, ids, weights, candidate=True)
        torch.cuda.synchronize()
        diff = (baseline.float() - candidate.float()).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        cosine = float(torch.nn.functional.cosine_similarity(
            baseline.float().flatten(), candidate.float().flatten(), dim=0
        ))
        scale = float(baseline.float().abs().max().clamp_min(1.0))
        bound = max(0.15, 0.08 * scale)
        if not torch.isfinite(candidate).all() or max_abs >= bound or cosine < 0.995:
            failed = True
        if args.graph:
            for _ in range(3):
                one_call(layer, x, ids, weights, candidate=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = one_call(layer, x, ids, weights, candidate=True)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, candidate, rtol=1e-4, atol=1e-4)
            original = x.clone()
            x.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(captured) == 0, "stale graph output"
            x.copy_(original)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, candidate, rtol=1e-4, atol=1e-4)
        baseline_ms = statistics.median(time_ms(
            layer, x, ids, weights, candidate=False, repeats=args.repeats
        ))
        candidate_ms = statistics.median(time_ms(
            layer, x, ids, weights, candidate=True, repeats=args.repeats
        ))
        print(
            f"{tokens:6d} {baseline_ms:11.4f} {candidate_ms:12.4f} "
            f"{baseline_ms / candidate_ms:7.3f} {max_abs:7.4f} "
            f"{mean_abs:8.5f} {cosine:7.5f}",
            flush=True,
        )
    if failed:
        raise SystemExit("EXL3 cooperative decode A/B: FAIL")
    print("EXL3 cooperative decode A/B: PASS", flush=True)


if __name__ == "__main__":
    main()
