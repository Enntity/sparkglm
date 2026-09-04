# Public report draft — do not publish automatically

## Scope caveat

These results are from one Atlas development branch, two DGX Sparks, and the
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` checkpoint. They prove concrete bugs
and scheduler behavior in this implementation. They do **not** prove that
every Mia, tdwild, or other online recipe contains the same defects.

The tested EXL3 checkpoint is not the root cause. EXL3 only stores the routed
expert matrices differently; the stalls and deadlock occurred in GLM's
prefill scheduling, EP command protocol, graph state, and DSA implementation.

## Issue draft

**Title:** GLM-5.3-Flash EXL3 on 2x DGX Spark: prefill must be bounded and EP-symmetric

### What users see

Start request A and let it stream. Submit a substantial prompt B. Both GPUs
remain busy, but A pauses for seconds and B takes a long time to emit its first
token. GPU utilization looks like concurrency, but the work is substantially
serialized: the device is spending a long phase advancing B rather than
decoding A.

### What we found

There were four separable implementation problems:

1. A generic safety guard made every MLA model's prefill monolithic. That is
   appropriate when only ordinary paged-MLA KV state exists, but GLM has a
   complete persistent latent cache, semantic pool, incomplete tail, and KDA
   state that can safely continue across bounded chunks.
2. Rank 0 could enter a private two-phase prefill chunk loop under expert
   parallelism while rank 1 waited for the next distributed command. This was
   an EP protocol deadlock, not slow quantized matrix multiplication.
3. Decode CUDA graphs captured pointers derived from transient host-side
   tables. Graph replay could later dereference invalid storage and report an
   illegal CUDA address.
4. Sparse DSA prefill launched causally per token and scored unused pool
   capacity. Prompt-batched causal kernels plus visible-complete-pool bounds
   remove that avoidable work.

The scheduler now admits GLM chunking explicitly, disables head-only two-phase
prefill under EP, runs the same ordinary commands on both ranks, and alternates
four decode ticks with one bounded prefill slab. Normal prefill may use 128
tokens; while a decode is active, a separate 32-token slab bounds streaming
pauses. Stable device pointer tables fix graph replay, and DSA now uses batched
causal prompt kernels and active-pool bounds.

### Why DS4F behaves differently

DeepSeek-V4-Flash is attention/MLA throughout. It does not have GLM's chain of
34 KDA recurrent layers, each of which updates a per-request FP32 matrix plus
causal convolution history. Atlas also began with mature family-specific
MLA/MoE/mHC/EP paths for DeepSeek. GLM therefore adds both a fixed-state cost
and a new scheduling/kernel path that DS4F does not pay in the same form.

This explains why DS4F can look much more concurrent in this stack; it does not
mean DS4F has no prefill contention, nor that GLM's roughly 145.6 MiB of fixed
KDA state per request caused this particular stall. The observed collapse was
execution scheduling and implementation work, not a state-capacity OOM.

### Simple reproduction

Start the two-rank endpoint, then run:

```bash
python3 bench/glm53/staggered_openai.py \
  --base-url http://127.0.0.1:8888 \
  --model GLM-5.3-Flash-EXL3 \
  --concurrency 2 --stagger-ms 8000 \
  --prompt-tokens 256 --output-tokens 64 --min-output-tokens 64
```

The fixture requested approximately 256 tokens but produced 498 with the
tested tokenizer. Keep the request bytes fixed. Compare active-decode prefill
slabs of 128 and 32 tokens and inspect `max_inter_token_gap_s`, not only average
tokens per second.

Measured on our two Sparks:

| Active slab | A maximum token gap | A effective rate | B TTFT | Wall |
|---:|---:|---:|---:|---:|
| 128 | 2.028 s | 2.83 tok/s | 9.32 s | 32.51 s |
| 32 | 0.843 s | 4.32 tok/s | 15.90 s | 32.38 s |

The smaller slab cut the visible freeze by 58% and improved A's effective rate
by 53%, but increased B's TTFT by 71%. Total wall time barely changed. This is
an honest latency tradeoff: bounded alternation, not simultaneous prefill and
decode or free throughput.

The final C4 canary completed every request in FIFO order. TTFT was 7.41,
18.33, 29.09, and 40.21 seconds; maximum token gaps stayed between 0.14 and
0.75 seconds; total wall was 45.06 seconds. Cancellation followed immediately
by slot reuse also passed, as did a native GLM tool call. Health remained ready
without an observed CUDA or NCCL asynchronous fault.

### What is still unproven

- authoritative reference-logit and agreed greedy-token parity;
- C8, long-context, mixed-length, endurance, and rank/node failure behavior;
- whether a collective-safe mixed EP batch can improve aggregate throughput
  without reintroducing large token gaps;
- MTP, prefix reuse, and vision.

## X post draft

**1/5** We reproduced why GLM-5.3-Flash EXL3 can feel non-concurrent on 2x DGX
Spark: GPU utilization stays high while a prompt prefill monopolizes execution,
pausing an existing stream. Busy GPUs are not the same thing as concurrent
user-visible progress.

**2/5** In our Atlas path this was not an EXL3/NVFP4 problem. We found a
monolithic-MLA scheduler guard, an EP head-only prefill deadlock, unsafe CUDA-
graph pointer lifetime, and avoidable per-token/full-capacity DSA prefill work.

**3/5** With identical 498-token prompts, changing only the active-decode
prefill slab from 128 to 32 tokens cut the existing stream's worst gap from
2.03s to 0.84s and raised its effective rate 2.83→4.32 tok/s. Incoming TTFT
rose 9.32→15.90s; total wall stayed ~32.4s. A tradeoff, not magic throughput.

**4/5** DS4F differs because it is MLA/attention throughout and has no 34-layer
KDA recurrent-state chain; its family-specific MLA/MoE/mHC/EP paths are also
more mature in this stack. That explains our result, but does not make a claim
about every GLM or DS4F recipe.

**5/5** The practical fix is GLM-aware chunking plus EP-symmetric commands and
a separate small prefill slab while decode is active. Text, native tool calls,
cancellation/slot reuse, and C4 now work. Next: reference parity, C8/soak/fault
tests, then a real collective-safe mixed EP batch. Repro and code should make
this improve quickly.
