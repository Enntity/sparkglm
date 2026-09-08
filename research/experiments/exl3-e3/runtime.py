# SPDX-License-Identifier: Apache-2.0
"""Opt-in large-prefill adapter; preserves SparkGLM thin and decode kernels."""
import os
import torch
import exl3_fat_moe_ext as ext
from sparkglm_e3_tables import build_grouped_fat_tables

_scratch = {}
_seen_policy = set()


def eligible(x, rows, intermediate):
    mode = os.environ.get('SPARKGLM_EXL3_E3_POLICY', 'large')
    if mode not in ('large', 'concurrent'):
        raise ValueError('E3 policy must be large or concurrent')
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from sparkglm_e3_policy import concurrent_prefill
    present = is_forward_context_available()
    metadata = get_forward_context().attn_metadata if present else {}
    profiling = present and metadata is None
    if profiling:
        # Account for persistent E3 buffers, then let the caller execute the
        # reference path too. Solo prefills need its separate scratch buffers;
        # profiling only E3 leaves those allocations out of the KV budget.
        _scratch_buffers(x, rows, intermediate)
        selected = False
    else:
        selected = mode == 'large' or concurrent_prefill(metadata)
    label = 'profile-reserve' if profiling else ('selected' if selected else 'reference')
    if os.environ.get('SPARKGLM_EXL3_E3_TRACE') == '1' and label not in _seen_policy:
        print(f'[sparkglm-e3-policy] {mode}: {label}', flush=True)
        _seen_policy.add(label)
    return selected


def _scratch_buffers(x, rows, intermediate):
    key = (str(x.device), x.shape[1], intermediate)
    old = _scratch.get(key)
    if old is None or old[0].shape[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm E3 maximum prefill shape before CUDA graph capture')
        old = (torch.empty((rows, x.shape[1]), device=x.device, dtype=torch.float16),
               torch.empty((rows, intermediate), device=x.device, dtype=torch.float16))
        _scratch[key] = old
    return old


def apply(x, out, counts, routes, weights, pointers, intermediate, cap, limit):
    rows = routes.numel()
    old = _scratch_buffers(x, rows, intermediate)
    h13, h2 = (t[:rows] for t in old)
    t = build_grouped_fat_tables(counts, cap, routes, weights, rows, 64)
    p = pointers
    ext.exl3_fat_moe_gather(x, t['row_token'], t['row_expert'], p['gate_suh'], h13, t['num_rows'])
    ext.exl3_fat_moe_gateup(h13, p['gate_trellis'], p['up_trellis'], p['gate_svh'],
                         p['up_svh'], p['down_suh'], h2, t['seg_expert'],
                         t['seg_row0'], t['seg_rows'], t['num_segs'], float(limit))
    ext.exl3_fat_moe_down(h2, p['down_trellis'], p['down_svh'], out, t['row_token'],
                       t['row_weight'], t['seg_expert'], t['seg_row0'], t['seg_rows'], t['num_segs'])
