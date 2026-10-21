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

A second patch, `flash_kda_tma_swizzle.patch`, applied after the slot patch,
moves the recurrence kernel's per-tile operands (V, the decayed and restored
keys and queries, and the 16x16 inverse and Mqk matrices) from interleaved to
128- and 32-byte swizzled shared-memory layouts. Each is then one or two TMA
boxes instead of dozens: the kernel's single load thread spent ~8.9K cycles a
16-token tile issuing them, and that bounded the recurrence. The values and
the MMAs are unchanged, only shared-memory addresses move: output and FP32
state are bit-identical to the unpatched library at T 2048/2051/4096/4100 and
H 16/32, and the TP-4 model's exact-prompt digest is unchanged. One 4096-token,
16-head call goes from 1.58 to 0.93 ms (recurrence 1.06 to 0.41 ms) on GB10.

A third patch, `flash_kda_fp32_recurrence.patch`, applied after the swizzle
patch, makes the chunked recurrence compute what the FP32 token recurrence
(Atlas decode and short prefill) computes. Upstream kept the recurrent state as
BF16 inside the kernel, rounded it every 16 tokens, used a tanh-approximated
sigmoid and an FP16 Neumann inverse, and rounded every operand to BF16, so the
same tokens prefilled to different hidden states depending on prompt length.
Now the state stays in FP32 registers for the whole call and is loaded and
stored as FP32; every MMA runs on split BF16 hi/lo operands with FP32
accumulation; the gate and beta logistic is exact; the inverse is an FP32
forward substitution; and v - k@S and the output sum are FP32. Kernel 2 owns
64 value columns per CTA while the grid fits one wave (TP-4's 16 heads), else
128 with eight math warps, and rebuilds the decayed operands from q, k and g
in prep warps. Kernel 1 now writes only INV, Mqk and the row normalizers, 2.2
instead of 13.5 KB a tile. Against an FP64 token recurrence on GB10 (H 16/32,
T 2048 to 16384, chunked calls handing the state over), the final state is
within 5.7e-6 relative (old library 6.7e-3, the FP32 recurrence 1.5e-6), and
99.4% of outputs equal the FP32 recurrence's BF16 output (old library 23%).
One call costs 1.06 instead of 0.92 ms at 4096 rows x 16 heads and 1.57
instead of 1.83 ms at 4100 x 32. Keeping the state FP32 between tiles follows
vllm-project/FlashKDA#13.

This closes the KDA compute path. Atlas still owns projections, scheduler state
lifecycle, layer assembly, and the model-level DSA/MLA integration.
