#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline synthetic Atlas/native sparse MLA comparison; no JIT or model loads."""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HEADS, DIM, CACHE, WIDTH = 32, 512, 16384, 2048
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
         scope="Synthetic fixed 2048-candidate comparison; 2051 candidates unhandled; no C4 or model-quality claim.")

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

    # Same BF16 source cache for both implementations. Generated directly in
    # BF16, avoiding a full-sized transient FP32 query at T4096.
    kv = torch.randn((CACHE, DIM), dtype=torch.bfloat16, device="cuda")
    packed_kv = pack(kv)
    table = torch.arange(CACHE // 16, device="cuda").to(torch.uint32)

    def fixture(rows):
        q = torch.randn((rows, HEADS, DIM), dtype=torch.bfloat16, device="cuda")
        # Unique, noncontiguous candidate sets per row. They span 16/64-token
        # pages and wrap the final physical page. Neither backend reselects IDs.
        r = torch.arange(rows, device="cuda", dtype=torch.int32)[:, None]
        c = torch.arange(WIDTH, device="cuda", dtype=torch.int32)[None, :]
        ids = ((r * 61 + c * 17 + 63) % CACHE).contiguous()
        lengths = torch.full((rows,), WIDTH, dtype=torch.int32, device="cuda")
        aout = torch.full_like(q, float("nan"))
        nout = torch.full_like(q, float("nan"))
        lse = torch.full((rows, HEADS), float("nan"), dtype=torch.float32, device="cuda")
        pq = padded(q)

        def run_atlas():
            status = atlas.atlas_sparse_run(q.data_ptr(), kv.data_ptr(), ids.data_ptr(), aout.data_ptr(), table.data_ptr(),
                               rows, HEADS, WIDTH, 16, scale, torch.cuda.current_stream().cuda_stream)
            check(status == 0, f"Atlas wrapper run failed with status {status}")

        def run_native():
            op(pq, packed_kv, ids, nout, lse, scale, 2, lengths, None, None, None, None)

        def run_inclusive():
            # Deliberately charge full 16K KV conversion plus all query padding
            # to every call. This is conservative versus an incremental cache.
            fresh_kv, fresh_q = pack(kv), padded(q)
            op(fresh_q, fresh_kv, ids, nout, lse, scale, 2, lengths, None, None, None, None)
            return fresh_kv, fresh_q  # retain allocation lifetimes through stop event

        run_atlas()
        run_native()
        torch.cuda.synchronize()
        check(aout.shape == nout.shape == q.shape, "output shape mismatch")
        check(bool(torch.isfinite(aout).all()) and bool(torch.isfinite(nout).all())
              and bool(torch.isfinite(lse).all()), "nonfinite or unwritten full output")
        samples = list(range(rows)) if rows == 128 else [0, 1, 63, 64, 127, 128, 2047, rows-1]
        ae, ne, lse_errors = [], [], []
        # Each row gathers only 2048*512 FP32 values (4MiB), never T*H*K*D.
        for row in samples:
            selected = kv[ids[row].long()].float()
            scores = (q[row].float() @ selected.T) * scale
            expected = scores.softmax(-1) @ selected
            a, n = error(aout[row], expected), error(nout[row], expected)
            check(a["relative_l2"] <= 0.015 and a["max_abs"] <= 0.04 and a["cosine"] >= 0.9995,
                  f"Atlas BF16 oracle failed row {row}: {a}")
            ae.append(a)
            ne.append(n)
            lse_errors.append(float((lse[row] - scores.logsumexp(-1)/math.log(2)).abs().max()))
        def summarize(records):
            return {"worst_row_relative_l2": max(v["relative_l2"] for v in records),
                    "max_abs": max(v["max_abs"] for v in records),
                    "min_row_cosine": min(v["cosine"] for v in records)}
        emit(stage="oracle", rows=rows, sampled_rows=samples, atlas_bf16_error=summarize(ae),
             native_bf16_drift=summarize(ne), native_bf16_lse_base2_drift=max(lse_errors),
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
