#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline 2051-candidate split-and-merge gate; no JIT or model loads."""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HEADS, DIM, CACHE, WIDTH = 32, 512, 16384, 2048
FULL_WIDTH = 2051
CAP, MIN_FREE = 2 * 1024**3, 4 * 1024**3


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    import tvm_ffi

    check(torch.cuda.is_available(), "CUDA unavailable")
    check(torch.cuda.get_device_capability() == (12, 1), "SM121 required")
    free, total = torch.cuda.mem_get_info()
    check(free >= MIN_FREE, f"requires 4GiB free; observed {free}")
    torch.cuda.set_per_process_memory_fraction(CAP / total)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    torch.manual_seed(20260911)
    torch.cuda.manual_seed_all(20260911)
    atlas = ctypes.CDLL(str(Path(args.atlas).resolve(strict=True)))
    atlas.atlas_sparse_init.argtypes, atlas.atlas_sparse_init.restype = [], ctypes.c_int
    atlas.atlas_sparse_run.argtypes = ([ctypes.c_void_p] * 5 + [ctypes.c_uint] * 4
                         + [ctypes.c_float, ctypes.c_void_p])
    atlas.atlas_sparse_run.restype = ctypes.c_int
    check(atlas.atlas_sparse_init() == 0, "Atlas wrapper init failed")
    native = tvm_ffi.load_module(str(Path(args.native).resolve(strict=True)))
    op = native.sparse_mla_sm120_paged_attention
    # Retained Atlas GLM effective_attn_scale(hd=256), not the 512 latent width.
    scale = 0.0625
    emit(stage="start", atlas_sha256=digest(args.atlas), native_sha256=digest(args.native),
         torch=torch.__version__, cuda=torch.version.cuda, free_bytes=free,
         torch_memory_cap_bytes=CAP, minimum_inclusive_speedup=1.3, attention_scale=scale,
         candidate_capacity=FULL_WIDTH, tail_counts=[0, 1, 2, 3],
         scope="Synthetic exact-ID split/merge; no C4 or model-quality claim.")

    def memcheck():
        peak = torch.cuda.max_memory_reserved()
        check(peak <= CAP, f"Torch peak reserved {peak} exceeds 2GiB")
        return peak

    def pack(kv):
        groups = kv.float().reshape(CACHE, 4, 128)
        scales = groups.abs().amax(-1).clamp_min(1e-4) / 448
        fp8 = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        packed = torch.zeros((CACHE, 656), dtype=torch.uint8, device="cuda")
        packed[:, :512] = fp8.contiguous().view(torch.uint8).reshape(CACHE, DIM)
        packed[:, 512:528] = scales.contiguous().view(torch.uint8).reshape(CACHE, 16)
        return packed.reshape(CACHE // 64, 64, 656)

    def padded(q):
        result = torch.zeros((q.shape[0], HEADS, 576), dtype=torch.bfloat16, device="cuda")
        result[..., :DIM] = q
        return result

    def error(actual, expected):
        a, e = actual.float(), expected.float()
        return {"relative_l2": float((a-e).norm()/e.norm().clamp_min(1e-12)),
                "max_abs": float((a-e).abs().max()),
                "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0))}

    def quantize_q_row(q):
        groups = q.float().reshape(HEADS, 4, 128)
        raw = groups.abs().amax(-1).clamp_min(1e-4) / 448
        bits = raw.contiguous().view(torch.int32)
        rounded = torch.where((bits & 0x007FFFFF) != 0,
                              (bits + 0x00800000) & 0x7F800000, bits)
        scales = rounded.contiguous().view(torch.float32)
        quant = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return (quant * scales[..., None]).reshape(HEADS, DIM)

    def decode_packed(packed):
        flat = packed.view(CACHE, 656)
        quant = flat[:, :512].contiguous().view(torch.float8_e4m3fn).float().reshape(CACHE, 4, 128)
        scales = flat[:, 512:528].contiguous().view(torch.float32)
        return (quant * scales[..., None]).reshape(CACHE, DIM)

    # Same BF16 source cache for both implementations. Generated directly in
    # BF16, avoiding a full-sized transient FP32 query at T4096.
    table = torch.arange(CACHE // 16, device="cuda").to(torch.uint32)

    def fixture(rows, simple=False):
        kv = (torch.full((CACHE, DIM), 0.5, dtype=torch.bfloat16, device="cuda") if simple else
              torch.randn((CACHE, DIM), dtype=torch.bfloat16, device="cuda"))
        packed_kv = pack(kv)
        decoded = decode_packed(packed_kv)
        q = torch.randn((rows, HEADS, DIM), dtype=torch.bfloat16, device="cuda")
        if simple:
            q.zero_()
        # Unique, noncontiguous candidate sets per row. They span 16/64-token
        # pages and wrap the final physical page. Neither backend reselects IDs.
        r = torch.arange(rows, device="cuda", dtype=torch.int32)[:, None]
        c = torch.arange(FULL_WIDTH, device="cuda", dtype=torch.int32)[None, :]
        ids = ((r * 61 + c * 17 + 63) % CACHE).contiguous()
        tail_counts = (r[:, 0] % 4).contiguous()
        ids[:, WIDTH:] = torch.where(torch.arange(3, device="cuda")[None, :] < tail_counts[:, None],
                                     ids[:, WIDTH:], -1)
        main_ids = ids[:, :WIDTH].contiguous()
        tail_ids = torch.full((rows, WIDTH), -1, dtype=torch.int32, device="cuda")
        tail_ids[:, :3] = ids[:, WIDTH:]
        # Zero-length native rows are not assumed safe. A scratch slot 0 call
        # for tail=0 is discarded by an EXACT zero merge weight; no Atlas ID is
        # replaced, appended, or removed. Main lengths always remain 2048.
        tail_ids[:, 0] = torch.where(tail_counts > 0, tail_ids[:, 0], 0)
        lengths = torch.full((rows,), WIDTH, dtype=torch.int32, device="cuda")
        tail_lengths = tail_counts.clamp_min(1)
        has_tail = (tail_counts > 0)[:, None]
        aout = torch.full_like(q, float("nan"))
        nout = torch.full_like(q, float("nan"))
        main_out, tail_out = torch.full_like(q, float("nan")), torch.full_like(q, float("nan"))
        lse = torch.full((rows, HEADS), float("nan"), dtype=torch.float32, device="cuda")
        main_lse, tail_lse = torch.full_like(lse, float("nan")), torch.full_like(lse, float("nan"))
        pq = padded(q)
        merge_scratch = torch.empty((rows, HEADS, DIM), dtype=torch.float32, device="cuda")

        def merge():
            # One reused FP32 buffer (256MiB at T4096), not multiple full-sized
            # casts/products and not a Python loop of thousands of GPU launches.
            l1 = torch.where(has_tail, tail_lse, -float("inf"))
            maximum = torch.maximum(main_lse, l1)
            w0, w1 = torch.exp2(main_lse - maximum), torch.exp2(l1 - maximum)
            total = w0 + w1
            merge_scratch.copy_(main_out)
            merge_scratch.mul_((w0 / total)[..., None])
            merge_scratch.addcmul_(tail_out, (w1 / total)[..., None])
            nout.copy_(merge_scratch)
            lse.copy_(maximum + torch.log2(total))

        def native_calls(query, cache):
            op(query, cache, main_ids, main_out, main_lse, scale, 2, lengths, None, None, None, None)
            op(query, cache, tail_ids, tail_out, tail_lse, scale, 2, tail_lengths, None, None, None, None)
            merge()

        def run_atlas():
            status = atlas.atlas_sparse_run(q.data_ptr(), kv.data_ptr(), ids.data_ptr(), aout.data_ptr(), table.data_ptr(),
                               rows, HEADS, FULL_WIDTH, 16, scale, torch.cuda.current_stream().cuda_stream)
            check(status == 0, f"Atlas wrapper run failed with status {status}")

        def run_native():
            native_calls(pq, packed_kv)

        def run_inclusive():
            # Deliberately charge full 16K KV conversion plus all query padding
            # to every call. This is conservative versus an incremental cache.
            fresh_kv, fresh_q = pack(kv), padded(q)
            native_calls(fresh_q, fresh_kv)
            return fresh_kv, fresh_q  # retain allocation lifetimes through stop event

        run_atlas()
        run_native()
        torch.cuda.synchronize()
        check(aout.shape == nout.shape == q.shape, "output shape mismatch")
        check(bool(torch.isfinite(aout).all()) and bool(torch.isfinite(nout).all())
              and bool(torch.isfinite(lse).all()), "nonfinite or unwritten full output")
        check(bool(torch.isfinite(main_out).all()) and bool(torch.isfinite(tail_out).all())
              and bool(torch.isfinite(main_lse).all()) and bool(torch.isfinite(tail_lse).all()),
              "nonfinite native partial output")
        if simple:
            expected_simple = torch.full_like(nout, 0.5)
            # The synthetic pack represents constant .5 to FP32 accuracy;
            # final BF16 output is exact. Test every mixed-tail row.
            check(torch.equal(nout, expected_simple), "zero-query constant KV must merge to exact .5")
            expected_lse = (WIDTH + tail_counts).float().log2()[:, None]
            check(float((lse - expected_lse).abs().max()) <= 0.0002, "zero-query merged base2 LSE mismatch")
        samples = list(range(rows)) if rows == 128 else [0, 1, 2, 3, 63, 64, 127, 128, 2047, rows-1]
        ae, ne, qe, lse_errors = [], [], [], []
        # Each row gathers only 2048*512 FP32 values (4MiB), never T*H*K*D.
        for row in samples:
            count = WIDTH + row % 4
            selected_ids = ids[row, :count].long()
            selected = kv[selected_ids].float()
            scores = (q[row].float() @ selected.T) * scale
            expected = scores.softmax(-1) @ selected
            a, n = error(aout[row], expected), error(nout[row], expected)
            check(a["relative_l2"] <= 0.015 and a["max_abs"] <= 0.04 and a["cosine"] >= 0.9995,
                  f"Atlas BF16 oracle failed row {row}: {a}")
            ae.append(a)
            ne.append(n)
            quant_kv = decoded[selected_ids]
            quant_scores = (quantize_q_row(q[row]) @ quant_kv.T) * scale
            quant_expected = quant_scores.softmax(-1) @ quant_kv
            quant_error = error(nout[row], quant_expected)
            check(quant_error["relative_l2"] <= 0.025 and quant_error["max_abs"] <= 0.04
                  and quant_error["cosine"] >= 0.9995,
                  f"native split/merge quantized oracle failed row {row}: {quant_error}")
            qe.append(quant_error)
            lse_error = float((lse[row] - quant_scores.logsumexp(-1)/math.log(2)).abs().max())
            check(lse_error <= 0.01, f"native split/merge base2 LSE mismatch row {row}: {lse_error}")
            lse_errors.append(lse_error)
        def summarize(records):
            return {"worst_row_relative_l2": max(v["relative_l2"] for v in records),
                    "max_abs": max(v["max_abs"] for v in records),
                    "min_row_cosine": min(v["cosine"] for v in records)}
        emit(stage="oracle", rows=rows, simple=simple, sampled_rows=samples, atlas_bf16_error=summarize(ae),
             native_bf16_drift=summarize(ne), native_quantized_error=summarize(qe),
             native_quantized_lse_base2_error=max(lse_errors),
             native_drift_is_report_only=True, torch_peak_reserved_bytes=memcheck())
        if rows == 128:
            return None
        functions = {"atlas": run_atlas, "native_prepacked": run_native, "native_inclusive": run_inclusive}
        for _ in range(3):
            for fn in functions.values():
                hold = fn()
                torch.cuda.synchronize()
                del hold
        timings = {name: [] for name in functions}
        names = list(functions)
        for pair in range(15):
            # Rotate execution order to avoid always rewarding the same warm cache.
            for name in names[pair % 3:] + names[:pair % 3]:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                hold = functions[name]()
                stop.record()
                stop.synchronize()
                timings[name].append(start.elapsed_time(stop))
                del hold
        check(bool(torch.isfinite(aout).all()) and bool(torch.isfinite(nout).all())
              and bool(torch.isfinite(lse).all()), "nonfinite output after timing")
        medians = {name: statistics.median(values) for name, values in timings.items()}
        inclusive_speedup = medians["atlas"] / medians["native_inclusive"]
        emit(stage="timing", rows=rows, median_ms=medians, samples_ms=timings,
             prepacked_speedup=medians["atlas"]/medians["native_prepacked"],
             inclusive_speedup=inclusive_speedup, candidate_pass=inclusive_speedup >= 1.3,
             torch_peak_reserved_bytes=memcheck())
        return inclusive_speedup

    fixture(128, simple=True)
    torch.cuda.empty_cache()
    fixture(128)
    torch.cuda.empty_cache()
    speedup = fixture(4096)
    check(speedup >= 1.3, "synthetic packing-inclusive speedup below 1.3; stop this candidate route")
    emit(stage="complete", passed=True, inclusive_speedup=speedup,
         scope="Synthetic speed candidate only; native BF16 drift reported, not waived or model-qualified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", default="/eval/libatlas_sparse.so")
    parser.add_argument("--native", default="/native/sparse_mla_sm120.so")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        emit(stage="failure", passed=False, error_type=type(exc).__name__, error=str(exc))
        sys.exit(1)
