#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure the exact BF16 TP=2 collective shapes used by GLM-5.3.

Run one process on each Spark with matching MASTER_ADDR/MASTER_PORT and RANK.
The benchmark intentionally uses torch.distributed/NCCL directly so it can be
run from the production image without loading model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 2048, 7104],
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def _time_cuda(
    operation: Callable[[], None], warmup: int, iterations: int
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _summarize(name: str, samples: list[float], tensor_bytes: int) -> dict:
    ordered = sorted(samples)
    median_ms = statistics.median(ordered)
    p10_ms = ordered[max(0, int(len(ordered) * 0.10) - 1)]
    p90_ms = ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))]
    # For a two-rank ring all-reduce, the algorithmic bus factor is one.
    gib_per_s = tensor_bytes / (median_ms / 1000) / (1024**3)
    return {
        "operation": name,
        "median_ms": round(median_ms, 4),
        "p10_ms": round(p10_ms, 4),
        "p90_ms": round(p90_ms, 4),
        "algorithmic_gib_s": round(gib_per_s, 3),
    }


def main() -> None:
    args = _parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    if world_size != 2:
        raise ValueError(
            "this appliance benchmark requires WORLD_SIZE=2, "
            f"got {world_size}"
        )

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print(
            json.dumps(
                {
                    "metadata": {
                        "world_size": world_size,
                        "dtype": "bfloat16",
                        "hidden_size": args.hidden_size,
                        "warmup": args.warmup,
                        "iterations": args.iterations,
                        "nccl_version": torch.cuda.nccl.version(),
                        "nccl_max_nchannels": os.environ.get("NCCL_MAX_NCHANNELS"),
                        "nccl_min_nchannels": os.environ.get("NCCL_MIN_NCHANNELS"),
                        "nccl_proto": os.environ.get("NCCL_PROTO"),
                        "nccl_algo": os.environ.get("NCCL_ALGO"),
                        "nccl_ib_qps_per_connection": os.environ.get(
                            "NCCL_IB_QPS_PER_CONNECTION"
                        ),
                        "nccl_ib_split_data_on_qps": os.environ.get(
                            "NCCL_IB_SPLIT_DATA_ON_QPS"
                        ),
                        "nccl_net_gdr_level": os.environ.get(
                            "NCCL_NET_GDR_LEVEL"
                        ),
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for tokens in args.tokens:
        shape = (tokens, args.hidden_size)
        tensor_bytes = tokens * args.hidden_size * torch.bfloat16.itemsize

        all_reduce_input = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")

        def all_reduce() -> None:
            dist.all_reduce(all_reduce_input)

        peer_rank = 1 - rank
        exchange_input = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        exchange_peer = torch.empty_like(exchange_input)
        exchange_sum = torch.empty_like(exchange_input)

        def exchange_add() -> None:
            operations = [
                dist.P2POp(dist.isend, exchange_input, peer_rank),
                dist.P2POp(dist.irecv, exchange_peer, peer_rank),
            ]
            for work in dist.batch_isend_irecv(operations):
                work.wait()
            torch.add(exchange_input, exchange_peer, out=exchange_sum)

        results = [
            _summarize(
                "all_reduce",
                _time_cuda(all_reduce, args.warmup, args.iterations),
                tensor_bytes,
            ),
            _summarize(
                "exchange_add",
                _time_cuda(exchange_add, args.warmup, args.iterations),
                tensor_bytes,
            ),
        ]

        exchange_check = torch.full(
            shape, rank + 1, dtype=torch.bfloat16, device="cuda"
        )
        exchange_check_peer = torch.empty_like(exchange_check)
        exchange_check_sum = torch.empty_like(exchange_check)
        reference = exchange_check.clone()
        dist.all_reduce(reference)
        check_operations = [
            dist.P2POp(dist.isend, exchange_check, peer_rank),
            dist.P2POp(dist.irecv, exchange_check_peer, peer_rank),
        ]
        for work in dist.batch_isend_irecv(check_operations):
            work.wait()
        torch.add(exchange_check, exchange_check_peer, out=exchange_check_sum)
        exchange_exact_tensor = torch.tensor(
            int(torch.equal(reference, exchange_check_sum)),
            device="cuda",
            dtype=torch.int32,
        )
        dist.all_reduce(exchange_exact_tensor, op=dist.ReduceOp.MIN)
        exchange_exact = bool(exchange_exact_tensor.item())

        rs_ag_exact: bool | None = None
        if tokens % world_size == 0:
            reduce_scatter_input = torch.zeros(
                shape, dtype=torch.bfloat16, device="cuda"
            )
            shard = torch.empty(
                (tokens // world_size, args.hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            gathered = torch.empty_like(reduce_scatter_input)

            def reduce_scatter() -> None:
                dist.reduce_scatter_single(shard, reduce_scatter_input)

            def all_gather() -> None:
                dist.all_gather_single(gathered, shard)

            def reduce_scatter_all_gather() -> None:
                dist.reduce_scatter_single(shard, reduce_scatter_input)
                dist.all_gather_single(gathered, shard)

            results.extend(
                [
                    _summarize(
                        "reduce_scatter",
                        _time_cuda(reduce_scatter, args.warmup, args.iterations),
                        tensor_bytes // 2,
                    ),
                    _summarize(
                        "all_gather",
                        _time_cuda(all_gather, args.warmup, args.iterations),
                        tensor_bytes // 2,
                    ),
                    _summarize(
                        "reduce_scatter_all_gather",
                        _time_cuda(
                            reduce_scatter_all_gather,
                            args.warmup,
                            args.iterations,
                        ),
                        tensor_bytes,
                    ),
                ]
            )

            # One correctness sample with non-zero rank-local partials. RS+AG
            # is mathematically identical to an all-reduce when no operation
            # is inserted while the sequence dimension is sharded.
            check = torch.full(
                shape, rank + 1, dtype=torch.bfloat16, device="cuda"
            )
            rs_reference = check.clone()
            dist.all_reduce(rs_reference)
            check_shard = torch.empty_like(shard)
            check_gathered = torch.empty_like(check)
            dist.reduce_scatter_single(check_shard, check)
            dist.all_gather_single(check_gathered, check_shard)
            exact = torch.equal(rs_reference, check_gathered)
            exact_tensor = torch.tensor(
                int(exact), device="cuda", dtype=torch.int32
            )
            dist.all_reduce(exact_tensor, op=dist.ReduceOp.MIN)
            rs_ag_exact = bool(exact_tensor.item())
            del (
                reduce_scatter_input,
                shard,
                gathered,
                check,
                rs_reference,
                check_shard,
                check_gathered,
                exact_tensor,
            )

        if rank == 0:
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "tensor_mib": round(tensor_bytes / (1024**2), 3),
                        "exchange_add_exact": exchange_exact,
                        "rs_ag_exact": rs_ag_exact,
                        "results": results,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        del (
            all_reduce_input,
            exchange_input,
            exchange_peer,
            exchange_sum,
            exchange_check,
            exchange_check_peer,
            exchange_check_sum,
            reference,
            exchange_exact_tensor,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
