# SPDX-License-Identifier: Apache-2.0
"""Opt-in large-prefill adapter; preserves SparkGLM thin and decode kernels."""
import torch
import exl3_fat_moe_ext as ext
from sparkglm_e3_tables import build_grouped_fat_tables

_scratch = {}


def apply(x, out, counts, routes, weights, pointers, intermediate, cap, limit):
    rows = routes.numel()
    key = (str(x.device), x.shape[1], intermediate)
    old = _scratch.get(key)
    if old is None or old[0].shape[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm E3 maximum prefill shape before CUDA graph capture')
        old = (torch.empty((rows, x.shape[1]), device=x.device, dtype=torch.float16),
               torch.empty((rows, intermediate), device=x.device, dtype=torch.float16))
        _scratch[key] = old
    h13, h2 = (t[:rows] for t in old)
    t = build_grouped_fat_tables(counts, cap, routes, weights, rows, 64)
    p = pointers
    ext.exl3_fat_moe_gather(x, t['row_token'], t['row_expert'], p['gate_suh'], h13, t['num_rows'])
    ext.exl3_fat_moe_gateup(h13, p['gate_trellis'], p['up_trellis'], p['gate_svh'],
                         p['up_svh'], p['down_suh'], h2, t['seg_expert'],
                         t['seg_row0'], t['seg_rows'], t['num_segs'], float(limit))
    ext.exl3_fat_moe_down(h2, p['down_trellis'], p['down_svh'], out, t['row_token'],
                       t['row_weight'], t['seg_expert'], t['seg_row0'], t['seg_rows'], t['num_segs'])
