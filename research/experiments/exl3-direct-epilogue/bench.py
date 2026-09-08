#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-free G1 gate; same script must run in reference and candidate images.

The unchanged serial M64 path is the arithmetic oracle, NOT the speed baseline.
Only compare grouped-to-grouped timings between identified images. These are
synthetic route distributions, not recorded production traces or G3 evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import exllamav3_ext as ext

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from exl3_grouped_prefill_ab import make_matrix, ptrs, HIDDEN, INTERMEDIATE


def run_case(count_list, seed, warmup, iterations, repeats, cap=128):
    device = torch.device('cuda')
    # The extension requires cap < input tokens even when every expert is thin.
    tokens = max(max(count_list), cap + 1)
    experts = len(count_list)
    torch.manual_seed(seed)
    x = (torch.randn(tokens, HIDDEN, device=device) * 0.05).half()
    counts = torch.tensor(count_list, dtype=torch.int64, device=device)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)))
    token_parts = [(torch.arange(n, device=device, dtype=torch.int64) + 17 * e) % tokens
                   for e, n in enumerate(count_list)]
    routes = torch.cat(token_parts)
    weights = torch.full((sum(count_list),), 1.0 / experts, dtype=torch.float16, device=device)
    matrices = []
    for e in range(experts):
        gate, suh, gate_svh = make_matrix(HIDDEN, INTERMEDIATE, seed + 100 + e)
        up, _, up_svh = make_matrix(HIDDEN, INTERMEDIATE, seed + 200 + e)
        down, down_suh, down_svh = make_matrix(INTERMEDIATE, HIDDEN, seed + 300 + e)
        matrices.append((gate, suh, gate_svh, up, up_svh, down, down_suh, down_svh))
    tables = tuple(ptrs([m[i] for m in matrices]) for i in range(8))
    total = sum(count_list)
    h13 = torch.empty(total, HIDDEN, dtype=torch.float16, device=device)
    gate_up = torch.empty(total, 2 * INTERMEDIATE, dtype=torch.float32, device=device)
    h2 = torch.empty(total, INTERMEDIATE, dtype=torch.float16, device=device)
    tasks = torch.empty((total + 63) // 64 + experts, 4, dtype=torch.int32, device=device)
    task_count = torch.empty(1, dtype=torch.int32, device=device)
    # Both output tensors have red zones; zero/copy operations touch only the view.
    guard = 12345.0
    storage = torch.full((tokens + 2, HIDDEN), guard, dtype=torch.float32, device=device)
    output = storage[1:-1]
    oracle = torch.zeros_like(output)
    slices = []
    begin = 0
    scratch_begin = 0
    for e, n in enumerate(count_list):
        if n > cap:
            m = matrices[e]
            end = begin + n
            h = x.index_select(0, routes[begin:end])
            transformed = torch.empty_like(h)
            gu = torch.empty(n, 2 * INTERMEDIATE, dtype=torch.float32, device=device)
            down_input = torch.empty(n, INTERMEDIATE, dtype=torch.float16, device=device)
            ext.had_r_128(h, transformed, m[1], None, 1.0)
            ext.exl3_fat_gemm_pair_m64(transformed, m[0], m[3], gu, m[2], m[4], 4, True, False)
            ext.exl3_fat_swiglu_had(gu, down_input, m[6], 10.0)
            ext.exl3_fat_gemm_scatter_m64(down_input, m[5], oracle, m[7],
                                        routes[begin:end], weights[begin:end], 4, True, False)
            # The planner packs only fat experts into intermediate buffers;
            # route offsets still include thin experts that this kernel skips.
            slices.append((scratch_begin, scratch_begin + n, transformed, gu, down_input))
            scratch_begin += n
        begin += n

    def grouped():
        output.zero_()
        ext.exl3_grouped_prefill_k4(x, output, offsets, routes, weights,
                                  h13, gate_up, h2, tasks, task_count, *tables, cap, 10.0)

    grouped()
    torch.cuda.synchronize()
    for begin, end, ref_h13, ref_gu, ref_h2 in slices:
        # No cross-expert atomics here: arithmetic must match exactly.
        assert torch.equal(h13[begin:end], ref_h13), 'gather/Hadamard drift'
        assert torch.equal(gate_up[begin:end], ref_gu), 'gate/up arithmetic drift'
        assert torch.equal(h2[begin:end], ref_h2), 'activation arithmetic drift'
    single = len(slices) <= 1

    def check():
        assert torch.isfinite(output).all(), 'non-finite output'
        assert torch.all(storage[0] == guard) and torch.all(storage[-1] == guard), 'output red-zone write'
        if single:
            assert torch.equal(output, oracle), 'single-expert output must be exact'
        else:
            # Same tolerance as the existing grouped gate; FP32 atomic ordering
            # is not deterministic when routes from several experts overlap.
            torch.testing.assert_close(output, oracle, rtol=1e-5, atol=1e-6)

    check()
    for _ in range(warmup):
        grouped()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped()
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        check()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    check()
    delta = (output - oracle).abs()
    return {'counts': count_list, 'tokens': tokens, 'cap': cap, 'seed': seed,
            'warmup': warmup, 'iterations': iterations, 'repeats': repeats,
            'grouped_graph_ms': samples, 'median_ms': statistics.median(samples),
            'max_abs': delta.max().item(), 'exact': torch.equal(output, oracle),
            'graph_replay_checked': True, 'red_zones_checked': True,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-id', required=True, help='actual image digest/ID, not a mutable tag')
    parser.add_argument('--source-sha256', required=True, help='installed exl3_fat_gemm.cu hash')
    parser.add_argument('--arm', choices=('reference', 'candidate'), required=True)
    parser.add_argument('--counts', help='one explicit comma-separated route list instead of the suite')
    parser.add_argument('--seed', type=int, default=20260905)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error('warmup, iterations and repeats must be positive')
    cases = ([[int(n) for n in args.counts.split(',')]] if args.counts else
             [[n] for n in (128, 129, 145, 192, 255, 256, 257, 383, 385, 512, 1024, 2048)] +
             [[0, 1, 128, 129, 255, 257, 512, 1024], [129, 192, 256, 383, 512, 640, 768, 1024]])
    if any(not c or min(c) < 0 or max(c) > 7168 or len(c) > 288 for c in cases):
        parser.error('counts require 1..288 experts and 0..7168 rows per expert')
    import re
    if not re.fullmatch(r'[0-9a-f]{64}', args.source_sha256):
        parser.error('source SHA-256 must contain 64 lowercase hex characters')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', args.image_id):
        parser.error('use the image ID returned by docker image inspect')
    installed_source = Path('/opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu')
    if not installed_source.is_file() or hashlib.sha256(installed_source.read_bytes()).hexdigest() != args.source_sha256:
        parser.error('installed source does not match the supplied SHA-256')
    torch.cuda.reset_peak_memory_stats()
    report = {'schema': 'sparkglm.direct-epilogue-g1/v1', 'arm': args.arm,
              'image_id': args.image_id, 'source_sha256': args.source_sha256,
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'extension_sha256': hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
              'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'timestamp_unix': time.time(), 'results': [],
              'limitations': ['Synthetic routes, not full model or production trace replay.',
                              'Warm weight reuse; does not reproduce full-model cache pressure.',
                              'Atomic overlapping-expert sums use declared tolerance, not bit-exactness.',
                              'No sanitizer run is implied by output red-zone checks.']}
    for counts in cases:
        print(f'Checking {args.arm}: counts={counts}', file=sys.stderr, flush=True)
        report['results'].append(run_case(counts, args.seed, args.warmup, args.iterations, args.repeats))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
