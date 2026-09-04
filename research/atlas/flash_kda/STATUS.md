<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# FlashKDA SM121 bridge

Atlas uses MoonshotAI's MIT-licensed FlashKDA inference kernels for the GLM-5.3
KDA chunk-prefill recurrence. The source is pinned to
`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`; its CUTLASS submodule is pinned to
`5c149f52a436782210263fb2f19b354443a61c6a`.

The upstream build list advertises SM90a, SM100a, SM103a, and SM120a. On a DGX
Spark GB10 with CUDA 13.1, the same source compiled unchanged for SM121f and
passed twelve exact upstream comparisons: fixed T64/H4, variable lengths
17/33/65 at H4, and batched B4/T256/H4, covering all four state I/O modes with
FP32 state.

`atlas_flash_kda_bridge.cu` removes the PyTorch dependency. It exposes an AOT C
ABI over Atlas device pointers and an in-place FP32 recurrent state whose
orientation is shared by prefill and decode. `rebuild.sh` reproduces the shared
library from the two pinned repositories; it does not fetch model weights.

Atlas adds a narrow source patch so FlashKDA maps each logical sequence to its
physical persistent-state slot inside the TMA state load/store. On GB10, a
varlen 17/33-token batch using fragmented slots `[3,1]` matched the untouched
upstream extension bit-for-bit for output and active FP32 state; inactive slots
were byte-identical. That avoids a 4 MiB-per-head-state gather/scatter at every
chunk boundary.

The Atlas-native prefill glue now covers the three depthwise conv4 + SiLU
streams, raw-beta transpose, and strict-FP32 gated RMSNorm. A separate GB10
probe passed with exact BF16 outputs and exact fragmented conv history. The AOT
binary, bridge, and slot patch are pinned in `PINS.sha256`.

This closes the KDA compute path. Atlas still owns projections, scheduler state
lifecycle, layer assembly, and the model-level DSA/MLA integration.
