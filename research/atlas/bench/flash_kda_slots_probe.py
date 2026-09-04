# SPDX-License-Identifier: AGPL-3.0-only

"""Prove Atlas's fragmented-slot FlashKDA bridge against upstream FlashKDA."""

import argparse
import ctypes
import math

import torch
import torch.nn.functional as F

import flash_kda


DIM = 128
LOWER_BOUND = -5.0


def device_pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def load_bridge(path: str) -> ctypes.CDLL:
    bridge = ctypes.CDLL(path)
    bridge.atlas_flash_kda_workspace_size.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    bridge.atlas_flash_kda_workspace_size.restype = ctypes.c_longlong
    bridge.atlas_flash_kda_prefill_fp32_state.argtypes = [
        *([ctypes.c_void_p] * 12),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_void_p,
    ]
    bridge.atlas_flash_kda_prefill_fp32_state.restype = ctypes.c_int
    return bridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    args = parser.parse_args()

    torch.manual_seed(53)
    device = torch.device("cuda")
    lengths = [17, 33]
    sequences = len(lengths)
    total_tokens = sum(lengths)
    heads = 4
    state_capacity = 4
    slots = torch.tensor([3, 1], dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor(
        [0, lengths[0], total_tokens], dtype=torch.int64, device=device
    )

    def normalized() -> torch.Tensor:
        values = torch.randn(
            (1, total_tokens, heads, DIM), dtype=torch.float32, device=device
        )
        return F.normalize(values, p=2, dim=-1).to(torch.bfloat16)

    query = normalized()
    key = normalized()
    value = torch.randn_like(query)
    forget = torch.randn_like(query)
    beta = torch.randn(
        (1, total_tokens, heads), dtype=torch.bfloat16, device=device
    )
    beta_ht = beta.view(total_tokens, heads).t().contiguous()
    a_log = torch.rand(heads, dtype=torch.float32, device=device)
    dt_bias = torch.rand(heads, DIM, dtype=torch.float32, device=device)
    query_scale = 1.0 / math.sqrt(DIM)

    state_pool = (
        torch.randn(
            (state_capacity, heads, DIM, DIM),
            dtype=torch.float32,
            device=device,
        )
        * 0.01
    )
    initial_pool = state_pool.clone()
    reference_initial = state_pool.index_select(0, slots.long()).clone()
    reference_final = torch.empty_like(reference_initial)
    reference_output = torch.empty_like(query)
    flash_kda.fwd(
        query,
        key,
        value,
        forget,
        beta,
        query_scale,
        reference_output,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=reference_initial,
        final_state=reference_final,
        cu_seqlens=cu_seqlens,
    )

    bridge = load_bridge(args.library)
    workspace_bytes = bridge.atlas_flash_kda_workspace_size(
        total_tokens, heads, sequences
    )
    upstream_workspace_bytes = flash_kda.get_workspace_size(
        total_tokens, heads, sequences
    )
    assert workspace_bytes == upstream_workspace_bytes
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    bridge_output = torch.empty_like(query).view(total_tokens, heads, DIM)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    status = bridge.atlas_flash_kda_prefill_fp32_state(
        device_pointer(query),
        device_pointer(key),
        device_pointer(value),
        device_pointer(forget),
        device_pointer(beta_ht),
        device_pointer(state_pool),
        device_pointer(bridge_output),
        device_pointer(workspace),
        device_pointer(a_log),
        device_pointer(dt_bias),
        device_pointer(cu_seqlens),
        device_pointer(slots),
        total_tokens,
        heads,
        sequences,
        state_capacity,
        query_scale,
        LOWER_BOUND,
        stream,
    )
    assert status == 0, f"bridge launch failed with CUDA status {status}"
    torch.cuda.synchronize()

    output_exact = torch.equal(
        bridge_output.view_as(reference_output), reference_output
    )
    active_state_exact = all(
        torch.equal(state_pool[slot], reference_final[sequence])
        for sequence, slot in enumerate(slots.tolist())
    )
    unused_slots = sorted(set(range(state_capacity)) - set(slots.tolist()))
    unused_state_exact = all(
        torch.equal(state_pool[slot], initial_pool[slot]) for slot in unused_slots
    )
    print(
        f"tokens={total_tokens} heads={heads} slots={slots.tolist()} "
        f"workspace={workspace_bytes} output_exact={output_exact} "
        f"active_state_exact={active_state_exact} "
        f"unused_state_exact={unused_state_exact}"
    )
    assert output_exact
    assert active_state_exact
    assert unused_state_exact


if __name__ == "__main__":
    main()
