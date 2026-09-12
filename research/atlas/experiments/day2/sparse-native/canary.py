#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline, build-free synthetic gate for the retained FlashInfer sparse MLA SO.

No model weights, FlashInfer imports, JIT, downloads, or service changes.
The parent must supply GPU isolation and an external wall-clock timeout.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


T, H, D, TOPK, PAGE, CACHE_TOKENS = 128, 32, 512, 2048, 64, 4096
MEMORY_LIMIT = 1024**3
ORACLE_TEMP_LIMIT = 32 * 1024**2
LENGTHS = [1, 3, 63, 64, 65, 127, 128, 129, 2048]


def report(**fields):
    print(json.dumps(fields, allow_nan=False), flush=True)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args):
    import torch
    import tvm_ffi

    library = Path(args.library).resolve(strict=True)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.get_device_capability() == (12, 1), "canary requires the retained SM121 device")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_LIMIT / total), 0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    torch.manual_seed(20260911)
    torch.cuda.manual_seed_all(20260911)
    device = torch.device("cuda:0")
    report(stage="start", library=str(library), sha256=digest, torch=torch.__version__,
           cuda=torch.version.cuda, shape=[T, H, D], topk=TOPK, lengths=LENGTHS,
           memory_limit_bytes=MEMORY_LIMIT, oracle_temp_limit_bytes=ORACLE_TEMP_LIMIT,
           warning="Retained binary provenance is not proof of parity with copied current source.")
    module = tvm_ffi.load_module(str(library))
    op = module.sparse_mla_sm120_paged_attention
    scale = 1 / math.sqrt(D)

    def memory_gate():
        reserved = torch.cuda.max_memory_reserved()
        require(reserved <= MEMORY_LIMIT, f"Torch reserved {reserved} bytes exceeds 1GiB")
        return reserved

    def pack(latent):
        # Synthetic arbitrary scales, not a claim to match vLLM's packer bytes.
        x = latent.float().reshape(CACHE_TOKENS, 4, 128)
        scales = x.abs().amax(-1).clamp_min(1e-4) / 448.0
        quant = (x / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        packed = torch.zeros((CACHE_TOKENS, 656), dtype=torch.uint8, device=device)
        packed[:, :512] = quant.contiguous().view(torch.uint8).reshape(CACHE_TOKENS, D)
        packed[:, 512:528] = scales.contiguous().view(torch.uint8).reshape(CACHE_TOKENS, 16)
        # Read back actual packed bytes for the oracle, including scale offsets.
        decoded_q = packed[:, :512].contiguous().view(torch.float8_e4m3fn).float()
        decoded_s = packed[:, 512:528].contiguous().view(torch.float32)
        decoded = (decoded_q.reshape(CACHE_TOKENS, 4, 128) * decoded_s[..., None]).reshape(CACHE_TOKENS, D)
        require(bool(torch.isfinite(decoded).all()), "packing produced nonfinite values")
        require(bool((packed[:, 528:] == 0).all()), "NoPE padding must remain zero")
        return packed.reshape(CACHE_TOKENS // PAGE, PAGE, 656), decoded

    def indices_for(lengths):
        ids = torch.full((T, TOPK), -1, dtype=torch.int32, device=device)
        # Odd step permutes 4096 slots, crossing pages and the final page boundary.
        starts = [0, 63, 64, 127, 128, CACHE_TOKENS - 1]
        for row, length in enumerate(lengths):
            chosen = (torch.arange(length, device=device) * 17 + starts[row % len(starts)]) % CACHE_TOKENS
            ids[row, :length] = chosen.to(torch.int32)
        return ids, torch.tensor(lengths, dtype=torch.int32, device=device)

    def quantized_query(q):
        x = q[..., :D].float().reshape(T, H, 4, 128)
        raw = x.abs().amax(-1).clamp_min(1e-4) / 448.0
        # Mirror the kernel's FP32 bitwise upward rounding to a power of two.
        bits = raw.contiguous().view(torch.int32)
        rounded = torch.where((bits & 0x007FFFFF) != 0,
                              (bits + 0x00800000) & 0x7F800000, bits)
        scales = rounded.contiguous().view(torch.float32)
        quant = (x / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return (quant * scales[..., None]).reshape(T, H, D)

    def oracle(q, kv, ids, lengths):
        # Per-row gather: no [T,H,TOPK,D] materialization. Conservative bound
        # includes two KV gathers, score/probability tensors, and head matrices.
        temp_bound = 2 * TOPK * D * 4 + 3 * H * TOPK * 4 + 4 * H * D * 4
        require(temp_bound <= ORACLE_TEMP_LIMIT, "oracle temporary-size bound exceeded")
        expected = torch.empty((T, H, D), dtype=torch.float32, device=device)
        expected_lse = torch.empty((T, H), dtype=torch.float32, device=device)
        for row, length in enumerate(lengths):
            selected = kv[ids[row, :length].long()]
            logits = (q[row] @ selected.T) * scale
            expected[row] = logits.softmax(-1) @ selected
            expected_lse[row] = logits.logsumexp(-1) / math.log(2)
        return expected, expected_lse

    def metrics(actual, expected):
        a, e = actual.float(), expected.float()
        diff = a - e
        return {"max_abs": float(diff.abs().max()),
                "relative_l2": float(diff.norm() / e.norm().clamp_min(1e-12)),
                "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0))}

    last = None
    for name in ("singleton", "zero_query", "random"):
        lengths = [1] * T if name == "singleton" else [LENGTHS[i % len(LENGTHS)] for i in range(T)]
        ids, lens = indices_for(lengths)
        if name == "random":
            latent = torch.randn((CACHE_TOKENS, D), device=device).to(torch.bfloat16)
            q = torch.zeros((T, H, 576), dtype=torch.bfloat16, device=device)
            q[..., :D] = torch.randn((T, H, D), device=device).to(torch.bfloat16)
        else:
            # Exact E4M3 cache with scale=1 is used below. The singleton varies
            # by physical token and channel; zero-query constant KV yields exact
            # constant output and log2(length), independently of FP8 QK rounding.
            if name == "singleton":
                latent = (((torch.arange(CACHE_TOKENS, device=device)[:, None]
                            + torch.arange(D, device=device)[None, :]) % 17 - 8) * 0.25).to(torch.bfloat16)
            else:
                latent = torch.full((CACHE_TOKENS, D), 0.5, dtype=torch.bfloat16, device=device)
            q = torch.zeros((T, H, 576), dtype=torch.bfloat16, device=device)
        kv, decoded = pack(latent)
        if name != "random":
            # Guarantee exact representability for the two first gates.
            flat = kv.view(CACHE_TOKENS, 656)
            flat[:, :512] = latent.to(torch.float8_e4m3fn).view(torch.uint8)
            flat[:, 512:528] = torch.ones((CACHE_TOKENS, 4), dtype=torch.float32, device=device).view(torch.uint8)
            decoded = latent.float()
        output = torch.full((T, H, D), float("nan"), dtype=torch.bfloat16, device=device)
        lse = torch.full((T, H), float("nan"), dtype=torch.float32, device=device)
        started = time.monotonic()
        op(q, kv, ids, output, lse, scale, 2, lens, None, None, None, None)
        torch.cuda.synchronize()
        require(tuple(output.shape) == (T, H, D) and tuple(lse.shape) == (T, H), "bad output shapes")
        require(bool(torch.isfinite(output).all()) and bool(torch.isfinite(lse).all()), f"{name}: nonfinite/unwritten output")
        if name == "singleton":
            expected = decoded[ids[:, 0].long()][:, None, :].expand(T, H, D)
            expected_lse = torch.zeros_like(lse)
            require(torch.equal(output, expected.to(torch.bfloat16)), "singleton must reproduce exact selected KV")
        elif name == "zero_query":
            expected = torch.full_like(output, 0.5).float()
            expected_lse = lens.float().log2()[:, None].expand(T, H)
            require(torch.equal(output, expected.to(torch.bfloat16)), "zero-query constant KV must remain exact")
        else:
            expected, expected_lse = oracle(quantized_query(q), decoded, ids, lengths)
        error = metrics(output, expected)
        lse_error = float((lse - expected_lse).abs().max())
        require(lse_error <= (0.01 if name == "random" else 0.0002), f"{name}: base2 LSE mismatch {lse_error}")
        if name == "random":
            require(error["relative_l2"] <= 0.025 and error["max_abs"] <= 0.04 and error["cosine"] >= 0.9995,
                    f"random quantized-oracle gate failed: {error}")
            original, _ = oracle(q[..., :D].float(), latent.float(), ids, lengths)
            drift = metrics(output, original)
        else:
            drift = None
        report(stage=name, passed=True, oracle_error=error, lse_max_abs=lse_error,
               original_bf16_drift=drift, elapsed_s=time.monotonic()-started,
               torch_peak_reserved_bytes=memory_gate())
        last = (q, kv, ids, output, lse, lens)

    q, kv, ids, output, lse, lens = last
    for _ in range(3):
        op(q, kv, ids, output, lse, scale, 2, lens, None, None, None, None)
    torch.cuda.synchronize()
    timings = []
    for _ in range(10):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        op(q, kv, ids, output, lse, scale, 2, lens, None, None, None, None)
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop))
    report(stage="complete", passed=True, kernel_ms_median=statistics.median(timings),
           kernel_ms_min=min(timings), timed_calls=len(timings),
           torch_peak_reserved_bytes=memory_gate(),
           scope="Synthetic mixed-length native prefill only; packing excluded; no Atlas speedup or end-to-end parity claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="/native/sparse_mla_sm120.so")
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as error:
        report(stage="failure", passed=False, error_type=type(error).__name__, error=str(error))
        sys.exit(1)
