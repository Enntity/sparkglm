#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare Atlas's raw EXL3 MoE ABI with a reconstructed-weight oracle.

Loads exactly one GLM routed expert (about 12.6 MB), not the full checkpoint.
Run inside the pinned Mia image after compiling exl3_raw_launcher.cu.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


HIDDEN = 4096
INTERMEDIATE = 2048
NUM_EXPERTS = 288
MAX_ROWS = 512
CONCURRENCY = 3
LOCK_INTS = 1024 * 1024 + 2 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--fat", action="store_true")
    parser.add_argument("--bench-iterations", type=int, default=0)
    return parser.parse_args()


def tensor_file(model: Path, name: str) -> Path:
    import json

    index = json.loads((model / "model.safetensors.index.json").read_text())
    return model / index["weight_map"][name]


def load_projection(model: Path, prefix: str) -> dict[str, torch.Tensor]:
    names = {suffix: f"{prefix}.{suffix}" for suffix in ("trellis", "suh", "svh", "mcg")}
    by_file: dict[Path, list[tuple[str, str]]] = {}
    for suffix, name in names.items():
        by_file.setdefault(tensor_file(model, name), []).append((suffix, name))
    loaded: dict[str, torch.Tensor] = {}
    for filename, entries in by_file.items():
        with safe_open(filename, framework="pt", device="cpu") as tensors:
            for suffix, name in entries:
                loaded[suffix] = tensors.get_tensor(name).cuda().contiguous()
    assert loaded["trellis"].dtype == torch.int16
    assert loaded["suh"].dtype == torch.float16
    assert loaded["svh"].dtype == torch.float16
    assert int(loaded["mcg"].view(torch.int32)[0]) == -877912083
    return loaded


def reconstruct_linear(
    x: torch.Tensor, projection: dict[str, torch.Tensor]
) -> torch.Tensor:
    import exllamav3_ext as ext

    rows, in_features = x.shape
    out_features = int(projection["svh"].numel())
    transformed = torch.empty_like(x, dtype=torch.float16)
    ext.had_r_128(x.half(), transformed, projection["suh"], None, 1.0)
    weight = torch.empty(
        (in_features, out_features), dtype=torch.float16, device=x.device
    )
    ext.reconstruct(weight, projection["trellis"], 4, True, False)
    output = (transformed.float() @ weight.float()).half()
    ext.had_r_128(output, output, None, projection["svh"], 1.0)
    assert output.shape == (rows, out_features)
    return output


def pointer_table(tensor: torch.Tensor, expert: int) -> torch.Tensor:
    pointers = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device="cuda")
    pointers[expert] = tensor.data_ptr()
    return pointers


def main() -> None:
    args = parse_args()
    if not 1 <= args.rows <= MAX_ROWS:
        raise SystemExit(f"--rows must be in [1, {MAX_ROWS}]")
    root = f"model.language_model.layers.{args.layer}.mlp.experts.{args.expert}"
    gate = load_projection(args.model, f"{root}.gate_proj")
    up = load_projection(args.model, f"{root}.up_proj")
    down = load_projection(args.model, f"{root}.down_proj")

    torch.manual_seed(53)
    x = torch.randn(args.rows, HIDDEN, dtype=torch.float16, device="cuda") * 0.25
    gate_ref = reconstruct_linear(x, gate)
    up_ref = reconstruct_linear(x, up)
    activation = F.silu(gate_ref).clamp(max=10.0) * up_ref.clamp(-10.0, 10.0)
    reference = reconstruct_linear(activation.half(), down).float()

    tables = []
    for projection in (gate, up, down):
        for suffix in ("trellis", "suh", "svh"):
            tables.append(pointer_table(projection[suffix], args.expert))

    counts = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int64, device="cuda")
    counts[args.expert] = args.rows
    token_sorted = torch.arange(args.rows, dtype=torch.int64, device="cuda")
    weight_sorted = torch.ones(args.rows, dtype=torch.float16, device="cuda")
    output = torch.zeros(args.rows, HIDDEN, dtype=torch.float32, device="cuda")

    library = ctypes.CDLL(str(args.library))
    if args.fat:
        transformed = torch.empty(
            (args.rows, HIDDEN), dtype=torch.float16, device="cuda"
        )
        gate_up = torch.empty(
            (args.rows, 2 * INTERMEDIATE), dtype=torch.float32, device="cuda"
        )
        counts[NUM_EXPERTS] = 1
        descriptors = torch.zeros((32, 4), dtype=torch.int32, device="cuda")
        descriptors[0] = torch.tensor(
            [args.expert, 0, args.rows, 0], dtype=torch.int32, device="cuda"
        )
        launch = library.atlas_glm53_exl3_fat_launch
        launch.argtypes = [ctypes.c_uint64] * 17 + [ctypes.c_int] * 4 + [ctypes.c_float]
        launch.restype = ctypes.c_int
        launch_args = (
            x.data_ptr(),
            transformed.data_ptr(),
            gate_up.data_ptr(),
            output.data_ptr(),
            *(table.data_ptr() for table in tables),
            counts.data_ptr(),
            descriptors.data_ptr(),
            token_sorted.data_ptr(),
            weight_sorted.data_ptr(),
            HIDDEN,
            INTERMEDIATE,
            NUM_EXPERTS,
            args.rows,
            10.0,
        )
        status = launch(*launch_args)
        if status:
            raise RuntimeError(f"Atlas EXL3 fat launch failed with cudaError_t={status}")
        torch.cuda.synchronize()
        delta = (output - reference).abs()
        gate_delta = (gate_up[:, :INTERMEDIATE] - gate_ref.float()).abs()
        up_delta = (gate_up[:, INTERMEDIATE:] - up_ref.float()).abs()
        import exllamav3_ext as ext

        down_input_ref = torch.empty_like(activation, dtype=torch.float16)
        ext.had_r_128(
            activation.half(), down_input_ref, down["suh"], None, 1.0
        )
        down_input_delta = (
            transformed[:, :INTERMEDIATE] - down_input_ref
        ).abs()
        result = {
            "path": "fat",
            "rows": args.rows,
            "gate_up_max": float(gate_up.abs().max()),
            "down_input_max": float(transformed[:, :INTERMEDIATE].abs().max()),
            "gate_max_abs": float(gate_delta.max()),
            "up_max_abs": float(up_delta.max()),
            "down_input_max_abs": float(down_input_delta.max()),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "max_reference": float(reference.abs().max()),
            "finite": bool(torch.isfinite(output).all()),
        }
        print(result)
        if not result["finite"] or result["max_abs"] > 0.08:
            raise SystemExit("EXL3 fat-kernel parity gate failed")
        if args.bench_iterations:
            for _ in range(2):
                output.zero_()
                launch(*launch_args)
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            for _ in range(args.bench_iterations):
                output.zero_()
                launch(*launch_args)
            finished.record()
            finished.synchronize()
            print(
                {
                    "path": "fat",
                    "iterations": args.bench_iterations,
                    "mean_ms": started.elapsed_time(finished)
                    / args.bench_iterations,
                }
            )
        return

    temp_g = torch.empty((CONCURRENCY, MAX_ROWS, HIDDEN), dtype=torch.float16, device="cuda")
    temp_u = torch.empty_like(temp_g)
    temp_ig = torch.empty(
        (CONCURRENCY, MAX_ROWS, INTERMEDIATE), dtype=torch.float16, device="cuda"
    )
    temp_iu = torch.empty_like(temp_ig)
    locks = torch.zeros(LOCK_INTS, dtype=torch.int32, device="cuda")
    launch = library.atlas_glm53_exl3_launch
    launch.argtypes = (
        [ctypes.c_uint64] * 18
        + [ctypes.c_int] * 6
        + [ctypes.c_float, ctypes.c_uint64]
    )
    launch.restype = ctypes.c_int
    pointer_args = [
        x.data_ptr(),
        temp_g.data_ptr(),
        temp_u.data_ptr(),
        temp_ig.data_ptr(),
        temp_iu.data_ptr(),
        output.data_ptr(),
        *(table.data_ptr() for table in tables),
        counts.data_ptr(),
        token_sorted.data_ptr(),
        weight_sorted.data_ptr(),
    ]
    launch_args = (
        *pointer_args,
        HIDDEN,
        INTERMEDIATE,
        NUM_EXPERTS,
        1,
        MAX_ROWS,
        CONCURRENCY,
        10.0,
        locks.data_ptr(),
    )
    status = launch(*launch_args)
    if status:
        raise RuntimeError(f"Atlas EXL3 launch failed with cudaError_t={status}")
    torch.cuda.synchronize()

    delta = (output - reference).abs()
    result = {
        "path": "persistent",
        "rows": args.rows,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_reference": float(reference.abs().max()),
        "finite": bool(torch.isfinite(output).all()),
    }
    print(result)
    if not result["finite"] or result["max_abs"] > 0.08:
        raise SystemExit("EXL3 raw-kernel parity gate failed")
    if args.bench_iterations:
        for _ in range(2):
            output.zero_()
            launch(*launch_args)
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        for _ in range(args.bench_iterations):
            output.zero_()
            launch(*launch_args)
        finished.record()
        finished.synchronize()
        print(
            {
                "path": "persistent",
                "iterations": args.bench_iterations,
                "mean_ms": started.elapsed_time(finished) / args.bench_iterations,
            }
        )


if __name__ == "__main__":
    main()
