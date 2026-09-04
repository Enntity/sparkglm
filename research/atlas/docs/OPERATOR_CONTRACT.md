# GLM-5.3-Flash operator contract

This is the implementation contract pinned to the official checkpoint revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0` and Transformers revision
`83d46aa2a2c47bef8350580f30981c974add0983`.

Primary sources:

- <https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json>
- <https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/model.safetensors.index.json>
- <https://github.com/huggingface/transformers/blob/83d46aa2a2c47bef8350580f30981c974add0983/src/transformers/models/glm5_next/modeling_glm5_next.py>
- <https://github.com/huggingface/transformers/blob/83d46aa2a2c47bef8350580f30981c974add0983/src/transformers/models/glm5_next/configuration_glm5_next.py>

## Base topology

- hidden width 4096, vocabulary 154,880, maximum position 1,048,576;
- 45 base layers, followed by physical MTP layer 45;
- KDA at every layer except `i % 4 == 3` (34 total);
- sparse NoPE-MLA at `[3,7,11,15,19,23,27,31,35,39,43]`;
- dense MLP at 0-2, sparse MoE at 3-44;
- 288 routed experts, top 8, one shared expert, width 2048;
- sigmoid router, correction bias for selection only, normalized selected
  weights multiplied by 2.5;
- four mHC residual streams, 20 Sinkhorn iterations, epsilon `1e-6`.

## KDA

Geometry is 64 heads of width 128. Q, K, and V have independent depthwise
causal convolutions of width four followed by SiLU. Q and K are L2-normalized
in FP32 and Q is scaled by `1/sqrt(128)`.

For each token and head:

```text
g = -5 * sigmoid(exp(A_log) * (f + dt_bias))
S = exp(g) * S
memory = k^T * S
delta = sigmoid(beta) * (v - memory)
S = S + outer(k, delta)
out = q^T * S
```

Then apply per-head RMSNorm, a learned sigmoid output gate, flatten 64x128, and
project 8192 to 4096.

Persistent state per layer/sequence:

- recurrent `S [64,128,128]` FP32;
- Q/K/V convolution histories, each `[64*128,3]` FP32.

Checkpoint, rollback, cancellation, migration, and slot reuse must move all
four parts atomically.

## Sparse NoPE-MLA

- Q latent 1536, KV latent 512;
- 64 expanded heads, QK width 256, V width 256;
- RoPE width zero and `mla_use_nope=true`;
- main cache should retain the normalized 512-wide latent, not expanded heads.

Semantic index:

1. Project the shared normalized Q latent into 32 index heads of width 128.
2. Project the original hidden state into one 128-wide key and affine-LayerNorm.
3. Produce 32 query-head weights and 128 channel-wise compression gates.
4. For each complete group of four tokens, add learned `[4,128]` APE, softmax
   over the four positions per channel, and pool to one 128-wide key.
5. Score visible pools with ReLU dot products, combine the 32 heads, select
   512 pools, and expand them to 2048 raw token indices.
6. Append the visible incomplete tail of up to three raw tokens.

The maximum logical selected width is 2051. Invalid `-1` entries, causal pool
visibility, and left-padding offsets are part of correctness.

Persistent index state is the completed pooled-key cache plus the current raw
key/gate tail and validity/position metadata. It is not ordinary paged KV.

## mHC

Residual state has shape `[batch,tokens,4,4096]`. Normalize the flattened 16,384
channels and project to 24 controls: 4 pre-collapse, 4 new-output, and 16 matrix
controls. Pre-collapse uses sigmoid, new-output uses twice sigmoid, and the 4x4
matrix is row-softmaxed then alternately column/row normalized for 20 Sinkhorn
iterations.

Apply this independently around attention and FFN. After layer 44, mean the
four streams, then apply final RMSNorm.

## Weight and precision ABI

Global prefix is `model.language_model.`. Embedding and untied LM head are
BF16 `[154880,4096]`; final norm is BF16 `[4096]`.

Important KDA tensors per layer:

- Q/K/V `[8192,4096]` BF16 and convolution `[8192,1,4]` BF16;
- forget low ranks `[128,4096]`, `[8192,128]`;
- `dt_bias [8192]` and `A_log [64]` FP32;
- beta `[64,4096]`;
- output gate `[128,4096]`, `[8192,128]`;
- output norm `[128]`, output projection `[4096,8192]`.

Important sparse-MLA/index tensors:

- Q-A `[1536,4096]`, Q-B `[16384,1536]`;
- KV-A `[512,4096]`, KV-B `[32768,512]`;
- output `[4096,16384]`;
- index query `[4096,1536]`, key `[128,4096]`, key LayerNorm weight/bias
  `[128]`, head weights `[32,4096]`, APE `[4,128]`, gates `[128,4096]`.

The checkpoint uses native E4M3 with 128x128 scaling, but many tensors are
explicitly BF16/FP32. Tensor headers and `modules_to_not_convert` are
authoritative; dtype must not be inferred from neighboring weights.

## MTP and serving protocols

Physical layer 45 has fused `eh_proj [4096,8192]`, its own ordinary residual
attention+MoE block, and a shared-head norm. It has no mHC tensors and reuses
the base embedding and LM head. Do not reuse the DeepSeek-V4 MTP layout.

Official EOS IDs are `[154820,154827,154829]`. Prompt serialization starts with
`[gMASK]<sop>`. Tool calls use Poolside-style `<tool_call>`, `<arg_key>`, and
`<arg_value>` tags. Text-only serving must preserve these before it can claim
OpenAI-compatible tool support.

The two-Spark bootstrap requires Atlas EP protocol v2 on both ranks. EP v1
addresses only one sequence and forces the effective batch size to one, so a
GLM concurrency recipe must fail startup rather than silently fall back.
