# SPDX-License-Identifier: MIT
# Copied from MiaAI-Lab 2c0ebe55a91ac8c0868cd6cba264e14bce93c66b.
# Copyright (c) 2026 Mia's AI Lab; see LICENSE.
import torch

def _excl_cumsum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inclusive = torch.cumsum(x, 0)
    return inclusive - x, inclusive


def build_grouped_fat_tables(
    counts: torch.Tensor,
    cap: int,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    rows_cap: int,
    tile_rows: int,
) -> dict[str, torch.Tensor]:
    """Device-side row/segment tables for the grouped fat kernels.

    Fat experts (count > cap) are laid out back to back in expert order in a
    fat-row buffer; each kernel CTA owns one `tile_rows` slice of one expert.
    Everything is computed with device ops on capacity-sized tensors and the
    kernels read the live `num_rows` / `num_segs`, so no host sync happens
    and the layer stays CUDA-graph capturable. `counts` excludes the
    invalid/nonlocal sentinel bucket, which the sort places after every
    real expert, so sentinel routes never enter a fat segment.
    """
    n_exp = int(counts.numel())
    device = counts.device
    fat_rows = torch.where(counts > cap, counts, torch.zeros_like(counts))
    row_off, row_cum = _excl_cumsum(fat_rows)
    sorted_off, _ = _excl_cumsum(counts)
    tiles = (fat_rows + (tile_rows - 1)) // tile_rows
    tile_off, tile_cum = _excl_cumsum(tiles)
    num_segs = tile_cum[-1:].to(torch.int32)
    num_rows = row_cum[-1:].to(torch.int32)
    max_segs = (rows_cap + tile_rows - 1) // tile_rows + n_exp
    seg = torch.arange(max_segs, device=device)
    e = torch.searchsorted(tile_cum, seg, right=True).clamp_(max=n_exp - 1)
    local_tile = seg - tile_off[e]
    seg_row0 = row_off[e] + local_tile * tile_rows
    seg_rows = torch.clamp(fat_rows[e] - local_tile * tile_rows, min=0, max=tile_rows)
    r = torch.arange(rows_cap, device=device)
    re = torch.searchsorted(row_cum, r, right=True).clamp_(max=n_exp - 1)
    src = (sorted_off[re] + (r - row_off[re])).clamp_(max=rows_cap - 1)
    return {
        "seg_expert": e.to(torch.int32),
        "seg_row0": seg_row0.to(torch.int32),
        "seg_rows": seg_rows.to(torch.int32),
        "num_segs": num_segs,
        "num_rows": num_rows,
        "row_expert": re.to(torch.int32),
        "row_token": token_sorted.index_select(0, src),
        "row_weight": weight_sorted.index_select(0, src),
    }
