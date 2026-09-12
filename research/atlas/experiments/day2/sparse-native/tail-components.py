#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One-shot timing attribution for the already-validated mixed-tail fixture."""
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
         scope="Diagnostic component timings only; no new correctness or candidate acceptance claim.")

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
        if rows != 4096:
            # Consume exactly the original fixture RNG draws, without rerunning
            # native calls or oracles for already-validated small fixtures.
            return
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

        def run_native():
            native_calls(pq, packed_kv)

        # Establish valid partial outputs before separately timing the merge.
        run_native()
        torch.cuda.synchronize()
        check(bool(torch.isfinite(main_out).all()) and bool(torch.isfinite(tail_out).all())
              and bool(torch.isfinite(nout).all()) and bool(torch.isfinite(lse).all()),
              "nonfinite output in previously validated fixture")

        def run_main():
            op(pq, packed_kv, main_ids, main_out, main_lse, scale, 2, lengths, None, None, None, None)

        def run_tail():
            op(pq, packed_kv, tail_ids, tail_out, tail_lse, scale, 2, tail_lengths, None, None, None, None)

        functions = {"main_native": run_main, "tail_native": run_tail, "torch_merge": merge}
        timings = {name: [] for name in functions}
        names = list(functions)
        for repeat in range(15):
            for name in names[repeat % 3:] + names[:repeat % 3]:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                functions[name]()
                stop.record()
                stop.synchronize()
                timings[name].append(start.elapsed_time(stop))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        elements = rows * HEADS * DIM
        emit(stage="components", rows=rows, head_count=HEADS, sampled_repeats=15,
             median_ms=medians, samples_ms=timings, sum_component_medians_ms=sum(medians.values()),
             torch_merge_fraction=medians["torch_merge"]/sum(medians.values()),
             current_merge_output_traffic_bytes=30*elements,
             fused_merge_minimum_output_traffic_bytes=6*elements,
             torch_peak_reserved_bytes=memcheck(),
             qualification="Previously validated fixture; timed components are diagnostic, not a paired speedup.")

    fixture(128, simple=True)
    torch.cuda.empty_cache()
    fixture(128)
    torch.cuda.empty_cache()
    fixture(4096)
    emit(stage="complete", diagnostic_complete=True,
         candidate_speedup_gate_unchanged=1.3,
         scope="No candidate acceptance or integration; full packing-inclusive paired gate remains required.")


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
