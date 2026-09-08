#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare E3 to SparkGLM grouped prefill, including both device planners."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import exllamav3_ext as ref
import exl3_fat_moe_ext as e3

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from exl3_grouped_prefill_ab import make_matrix, ptrs, HIDDEN, INTERMEDIATE
from tables import build_grouped_fat_tables


def case(count_list, seed, iterations, repeats, cap):
    device = 'cuda'
    torch.manual_seed(seed)
    tokens = max(max(count_list), cap + 1)
    n_exp, total = len(count_list), sum(count_list)
    x = (torch.randn(tokens, HIDDEN, device=device) * 0.05).half()
    counts = torch.tensor(count_list, dtype=torch.int64, device=device)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)))
    routes = torch.cat([(torch.arange(n, device=device) + e * 17) % tokens
                        for e, n in enumerate(count_list)]).long()
    weights = torch.full((total,), 1 / n_exp, dtype=torch.float16, device=device)
    matrices = []
    for e in range(n_exp):
        g, suh, gsvh = make_matrix(HIDDEN, INTERMEDIATE, seed + 100 + e)
        u, _, usvh = make_matrix(HIDDEN, INTERMEDIATE, seed + 200 + e)
        d, dsuh, dsvh = make_matrix(INTERMEDIATE, HIDDEN, seed + 300 + e)
        matrices.append((g, suh, gsvh, u, usvh, d, dsuh, dsvh))
    p = tuple(ptrs([m[i] for m in matrices]) for i in range(8))
    h13 = torch.empty(total, HIDDEN, dtype=torch.float16, device=device)
    gu = torch.empty(total, 2 * INTERMEDIATE, dtype=torch.float32, device=device)
    h2 = torch.empty(total, INTERMEDIATE, dtype=torch.float16, device=device)
    eh13, eh2 = torch.empty_like(h13), torch.empty_like(h2)
    tasks = torch.empty((total + 63) // 64 + n_exp, 4, dtype=torch.int32, device=device)
    ntasks = torch.empty(1, dtype=torch.int32, device=device)
    guard = 12345.0
    rstorage = torch.full((tokens + 2, HIDDEN), guard, device=device)
    estorage = torch.full_like(rstorage, guard)
    rout, eout = rstorage[1:-1], estorage[1:-1]

    def reference():
        rout.zero_()
        ref.exl3_grouped_prefill_k4(x, rout, offsets, routes, weights, h13, gu, h2,
                                  tasks, ntasks, *p, cap, 10.0)

    def candidate():
        eout.zero_()
        t = build_grouped_fat_tables(counts, cap, routes, weights, total, 64)
        e3.exl3_fat_moe_gather(x, t['row_token'], t['row_expert'], p[1], eh13, t['num_rows'])
        e3.exl3_fat_moe_gateup(eh13, p[0], p[3], p[2], p[4], p[6], eh2,
                             t['seg_expert'], t['seg_row0'], t['seg_rows'], t['num_segs'], 10.0)
        e3.exl3_fat_moe_down(eh2, p[5], p[7], eout, t['row_token'], t['row_weight'],
                           t['seg_expert'], t['seg_row0'], t['seg_rows'], t['num_segs'])

    def check():
        torch.cuda.synchronize()
        for storage in (rstorage, estorage):
            assert torch.all(storage[0] == guard) and torch.all(storage[-1] == guard)
        assert torch.isfinite(eout).all() and torch.isfinite(rout).all()
        torch.testing.assert_close(eout, rout, rtol=1e-4, atol=1e-5)
        return float((eout - rout).abs().max())

    reference(); candidate()
    error = check()
    graphs = []
    for fn in (reference, candidate):
        for _ in range(10): fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): fn()
        graphs.append(graph)
    # Replay with changed activations and changed route weights, not stale buffers.
    x.mul_(0.7)
    weights.mul_(0.8)
    for g in graphs: g.replay()
    changed_error = check()
    changed_reference = rout.clone()
    reference(); candidate()
    check()
    torch.testing.assert_close(rout, changed_reference, rtol=1e-5, atol=1e-6)
    samples = [[], []]
    for repetition in range(repeats):
        for arm in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations): graphs[arm].replay()
            end.record(); end.synchronize()
            samples[arm].append(start.elapsed_time(end) / iterations)
    check()
    return dict(counts=count_list, cap=cap, tokens=tokens, seed=seed,
                reference_ms=samples[0], candidate_ms=samples[1],
                speedup=statistics.median(samples[0])/statistics.median(samples[1]),
                max_abs=error, changed_graph_max_abs=changed_error,
                reference_scratch_bytes=sum(t.numel()*t.element_size() for t in (h13,gu,h2,tasks,ntasks)),
                candidate_activation_bytes=sum(t.numel()*t.element_size() for t in (eh13,eh2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image-id', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=20260907)
    ap.add_argument('--iterations', type=int, default=40)
    ap.add_argument('--repeats', type=int, default=7)
    args = ap.parse_args()
    if not args.image_id.startswith('sha256:') or len(args.image_id) != 71:
        ap.error('image-id must be a complete immutable image ID')
    output = Path(args.output)
    if output.exists(): ap.error('output already exists')
    report = dict(schema='sparkglm.e3-g1/v1', image_id=args.image_id,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  timestamp_unix=time.time(), seed=args.seed, iterations=args.iterations,
                  repeats=args.repeats, results=[], status='running',
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*') if p.is_file()},
                  extension_sha256={name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                    for name,m in [('reference',ref),('e3',e3)]})
    cases = [([n],128) for n in (128,129,145,192,255,256,257,383,385,512,1024,2048)]
    cases += [([0,1,128,129,255,257,512,1024],128),
              ([129,192,256,383,512,640,768,1024],128),
              ([256]*16,128),([512]*32,128),([192]*288,128)]
    try:
        for counts,cap in cases:
            result=case(counts,args.seed,args.iterations,args.repeats,cap)
            report['results'].append(result)
            print(json.dumps({'counts':counts[:16], 'experts':len(counts),'speedup':result['speedup']}),flush=True)
            output.write_text(json.dumps(report,indent=2)+'\n')
        report['status']='passed-correctness'
    except Exception as exc:
        report['status']='failed'; report['error']=str(exc)
        raise
    finally:
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()
