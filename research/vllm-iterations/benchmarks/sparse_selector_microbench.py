#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness and latency probe for GLM-5.3's native sparse selector.

The GLM-5.3 indexer pools four KV positions into one score column. Therefore
16K and 32K token contexts exercise 4K and 8K score columns respectively.
This benchmark intentionally operates on the materialized score matrix so it
isolates the native top-k stage; scorer+selector fusion is a separate target.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Result:
    dtype: str
    rows: int
    columns: int
    top_k: int
    matrix_mib: float
    mean_us: float
    p50_us: float
    p95_us: float


def _percentile(samples: torch.Tensor, q: float) -> float:
    return float(torch.quantile(samples, q).item())


def _validate_topk_values(
    logits: torch.Tensor, indices: torch.Tensor, top_k: int
) -> None:
    if indices.dtype != torch.int32:
        raise AssertionError(f"expected int32 indices, got {indices.dtype}")
    if torch.any(indices < 0) or torch.any(indices >= logits.shape[1]):
        raise AssertionError("selector returned an out-of-range index")

    selected = torch.gather(logits, 1, indices.long()).sort(dim=1).values
    reference = logits.topk(top_k, dim=1).values.sort(dim=1).values
    if not torch.equal(selected, reference):
        mismatch = int(torch.count_nonzero(selected != reference).item())
        raise AssertionError(
            f"selected values differ from torch.topk at {mismatch} positions"
        )


@torch.inference_mode()
def run_case(
    *,
    rows: int,
    columns: int,
    top_k: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> Result:
    torch.manual_seed(17)
    logits = torch.randn((rows, columns), device="cuda", dtype=dtype)
    row_starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    row_ends = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    indices = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)

    def invoke() -> None:
        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            indices,
            rows,
            logits.stride(0),
            logits.stride(1),
            top_k,
        )

    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize()
    _validate_topk_values(logits, indices, top_k)

    elapsed = torch.empty(iterations, dtype=torch.float64)
    for iteration in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        elapsed[iteration] = start.elapsed_time(end) * 1000.0

    return Result(
        dtype=str(dtype).removeprefix("torch."),
        rows=rows,
        columns=columns,
        top_k=top_k,
        matrix_mib=round(logits.numel() * logits.element_size() / 2**20, 3),
        mean_us=round(float(elapsed.mean().item()), 3),
        p50_us=round(_percentile(elapsed, 0.50), 3),
        p95_us=round(_percentile(elapsed, 0.95), 3),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--columns", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    # SparkGLM uses vLLM's stable-libtorch extension. Importing it registers
    # the selector under torch.ops._C.* just like the conventional _C module.
    import vllm._C_stable_libtorch  # noqa: F401, PLC0415

    results = []
    for rows in args.rows:
        for columns in args.columns:
            for dtype in (torch.float32, torch.float16):
                result = run_case(
                    rows=rows,
                    columns=columns,
                    top_k=args.top_k,
                    dtype=dtype,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                results.append(asdict(result))
                print(json.dumps(results[-1], sort_keys=True), flush=True)

    print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
