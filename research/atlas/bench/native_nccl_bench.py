#!/usr/bin/env python3
"""Two-rank native NCCL all-reduce latency probe for the GLM verifier sizes."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--port", type=int, default=29653)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--bytes",
        type=int,
        nargs="+",
        default=[73_728, 147_456, 294_912],
        dest="payload_bytes",
    )
    return parser.parse_args()


def time_all_reduce(payload_bytes: int, warmup: int, iterations: int) -> float:
    if payload_bytes % 2:
        raise ValueError("BF16 payload byte count must be even")
    tensor = torch.ones(payload_bytes // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MASTER_ADDR", args.master)
    os.environ.setdefault("MASTER_PORT", str(args.port))
    dist.init_process_group("nccl", rank=args.rank, world_size=2)
    torch.cuda.set_device(0)
    try:
        for payload_bytes in args.payload_bytes:
            latency_ms = time_all_reduce(payload_bytes, args.warmup, args.iterations)
            if args.rank == 0:
                print(f"native_nccl bytes={payload_bytes} latency_ms={latency_ms:.4f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
