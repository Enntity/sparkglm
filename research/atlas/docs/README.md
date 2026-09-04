# GLM-5.3-Flash on Atlas

## Status

This branch implements the text-only, fixed two-Spark TP=2 serving path for
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`. The Rust workspace builds without
CUDA, and the production CUDA sources compile for `sm_121f`. Small GB10 probes
have exercised KDA recurrence and convolution glue, sparse DSA/indexing, the
FlashKDA fragmented-state bridge, a real 32 MiB DSA projection, the FP32 router,
and one real 12.6 MiB EXL3 expert. The full checkpoint now loads on two DGX
Sparks and passes text, tool-call, cancellation/slot-reuse, and C1/C2/C4
endpoint canaries. The optimized path uses EXL3 target weights, DFlash2 with
an eight-row verify block, pure TP=2, and one cross-sequence weight sweep for
both target verify and draft proposal. Authoritative reference-logit parity, long-context
endurance, node-failure recovery, prefix reuse, and vision are not yet
validated, so the measured receipts below are narrower than a production
readiness claim. Current TP=2 receipts are in
[`NATIVE_TP2_VALIDATION_2026-09-01.md`](../../../results/legacy/2026-09-01-atlas-native-tp2-validation/NATIVE_TP2_VALIDATION_2026-09-01.md).

Source contract:

- checkpoint revision: `04c4e9e95c5da8862dced7e5056455116f83a7e0`;
- Transformers reference revision: `83d46aa2a2c47bef8350580f30981c974add0983`;
- 45 base layers: 34 KDA and 11 sparse NoPE-MLA;
- layers 0-2 use dense MLPs; layers 3-44 use 288 experts, top 8, plus one
  shared expert;
- one MTP layer is physically stored as layer 45;
- mHC uses four streams and 20 Sinkhorn iterations;
- the target checkpoint uses EXL3 only for routed expert gate/up/down matrices;
- dense layers, the shared expert, KDA/DSA, hyperconnections, and router weights
  remain BF16/F32, and router logits must remain FP32 through top-k;
- NVFP4 is a future sibling expert backend, not a compatibility label for this
  checkpoint.

## The contention problem, simply

Ordinary attention keeps a token cache. GLM also keeps a large running matrix
for every live request in every KDA layer.

For one KDA layer and one request:

```text
64 heads x 128 key dims x 128 value dims x 4 FP32 bytes = 4 MiB
```

Across 34 KDA layers that is 136 MiB. The three width-4 causal convolutions
retain another 9.56 MiB of history. A live request therefore owns about
145.56 MiB of fixed KDA state before its context cache, temporary activations,
MoE workspace, or CUDA graphs are counted.

At 32,768 tokens, the BF16 sparse-MLA latent and pooled semantic index add about
374 MiB. The conservative persistent total is about 520 MiB per sequence.
This is why “max concurrent requests” cannot be a request-count or KV-only
setting for GLM.

## Why DeepSeek-V4-Flash does not hit the same wall

DeepSeek-V4-Flash is attention/MLA throughout. It has context-dependent cache
and its own compressor/index state, but it does not allocate GLM's 34 recurrent
`64x128x128` FP32 matrices for every sequence. Atlas already has dedicated MLA,
MoE, mHC, FP8, and EP machinery for that family, so it can batch requests over
shared layer weights without first paying GLM's roughly 146 MiB fixed KDA tax
per request.

That explains the architectural difference. It does not prove every observed
latency difference: scheduler policy, kernel quality, prompt lengths, context
limits, and recipe flags still need a controlled two-Spark comparison.

## What was added

| Layer | Added now | Safety consequence |
|---|---|---|
| Config | Strict nested GLM parser and exact layer/MLP/index contracts | Similar-but-wrong variants fail startup |
| Capability | Dedicated KDA and sparse-NoPE attention identities | GLM cannot silently route through GDN/ordinary MLA |
| Quantization | First-class `exl3` target and raw I16/F16/F32 tensor loading | Only routed experts are dequantized; native BF16/F32 tensors remain native |
| Checkpoint preflight | Exact KDA/DSA/dense/MoE/MTP names plus rank-local expert inventory | Truncated conversions and remote-expert materialization fail with the offending tensor |
| State | Checked KDA, latent-cache, pooled-index, and tail-state byte model | Admission can be based on bytes |
| EOS | Preserves all three official stop IDs | User/tool/observation endings are not lost |
| Prefix cache | KV-only prefix reuse disabled | No resume from stale KDA/index state |
| Model factory | Complete mixed-dtype GLM loader with rank-local EXL3 pointer tables | Wrong shapes/dtypes and remote-expert materialization fail before execution |
| Layer ABI | Separate KDA recurrent and Q/K/V convolution pointers | State layout no longer lies about GDN compatibility |
| Sparse state ABI | Latent, pooled-key, raw-tail, gate, and metadata pointers form one image | Cache/index cannot be restored independently |
| Scheduler | Explicit bounded decode/prefill cadence, shared by admission and continued chunks | EP ranks see the same ordinary commands; no hidden mixed forward |
| Overlap slabs | Separate full-prefill and active-decode slab limits | Long prompts make bounded progress without multi-second decode freezes |
| EP protocol | Head-only two-phase prefill disabled under EP | Rank 0 cannot enter a private chunk loop while rank 1 waits for commands |
| Decode graphs | Stable per-layer device pointer tables | Replayed graphs do not dereference transient host table storage |
| DSA prefill | Eight-query causal tiles plus a tensor-core sparse suffix | Removes the per-token launch loop while preserving the legacy rollback path |
| Tool protocol | Native GLM template mapped to Poolside-v1 parsing | Checkpoint-native tool calls reach the OpenAI response surface correctly |
| Runtime contract | Fixed EP=1/TP=2, batch four, DFlash2 eight-row blocks, prefix/swap off | The optimized recipe cannot silently fall back to the old serialized EP path |
| Memory preflight | Fixed KDA plus context-index pool, including an isolated collective padding slot | `max_batch_size` reserves the state it claims before loading weights |
| Numerical oracle | KDA, FlashKDA slots, DSA/index, BF16 DSA projection, FP32 router, and EXL3 expert probes | CUDA paths have small independent references without loading the whole model |
| GB10 target | Exact `(glm-5.3-flash, exl3)` identity | EXL3 does not masquerade as FP8 or NVFP4 |
| Router | BF16 gate matrix, FP32 GEMM output, FP32 sigmoid/correction/top-k | Matches `moe_router_dtype=float32`; avoids expert-selection flips from BF16 logit stores |
| KDA | Batched projections, conv4/SiLU glue, pinned FlashKDA prefill, fused decode, gated norm | Prefill and decode update the same fragmented FP32 state pool |
| DSA | BF16 latent projection, learned BF16 pooling, top-512 expansion, sparse NoPE-MLA | The semantic index and latent cache advance causally with each token |
| TP FFN | All 288 EXL3 experts on each rank with 2048 intermediate split to 1024 per rank | No expert dispatch; both ranks follow one identical TP collective order |
| Cross-sequence target | Sequence-major `n*K` verification across KDA, DSA, dense, and MoE layers | One target weight sweep serves up to four isolated request states |
| Cross-sequence drafter | Sequence-major DFlash2 QKV/conv/MLP/head with per-sequence paged attention and histories | Removes serialized proposer passes without request-state leakage |
| Adaptive depth experiment | Runtime-sized N=1/N=2/N=3/N=4 DFlash graphs | A 17-row ceiling can demote weak streams immediately; production starts at the measured K=8 optimum |
| Load probe | Dependency-free staggered OpenAI request harness | TTFT/decode regressions are measured per request, not inferred from GPU use |

## Remaining end-to-end pieces

1. Authoritative reference-logit and greedy-token parity on agreed golden
   prompts. Endpoint determinism is not a substitute for an external oracle.
2. C8, long-context, mixed-length, endurance, cancellation-soak, and node/rank
   failure recovery tests.
3. State-aware admission beyond the conservative fixed four-slot recipe:
   `min(KV capacity, KDA slots, index capacity, scratch, batch slots)`.
4. Complete KV + KDA + semantic-index snapshots before enabling prefix reuse.
   The fixed DFlash2 verification path has its own rollback/commit boundary.
5. Vision after text/tool acceptance. Audio must fail explicitly because this
   checkpoint has no audio encoder.

## Effort and expected return

| Change | Layer | Effort | Expected return |
|---|---|---:|---|
| Strict config/state contract | Core | Small | Prevents silent wrong-model execution; done |
| Decode/prefill interleaving | Scheduler | Small-medium | Large p95 TTFT improvement under mixed load; modest throughput change |
| KV + KDA + index admission | Scheduler/state | Medium | Prevents GPU thrash/OOM and makes concurrency predictable |
| Rank-local load before materialization | Loader/runtime | Medium-high | Required to fit the 328 GB checkpoint on two 128 GB Sparks |
| Native KDA kernels | GB10 kernels/layer | High | Required correctness; largest decode-path opportunity |
| Sparse NoPE-MLA/indexer | GB10 kernels/cache | High | Required correctness; avoids dense-attention fallback cost |
| Cross-sequence TP target and DFlash2 batches | Model/scheduler | High | Largest measured concurrency gain; done for C2-C4 |
| Device-compacted large-M EXL3 path | GB10 kernels/model | High | 4.0-6.0x one-expert microbench at 129-512 rows; prefill-only benefit |
| Complete prefix snapshots | State/cache | High | Better repeat-turn TTFT after parity |
| Vision | Processor/model | High | Adds multimodal capability; little text-serving ROI |

Scheduler interleaving alone should let short requests start between chunks of
a long prefill, so queued TTFT should improve materially. It cannot increase the
number of requests whose persistent state fits, and Atlas's current mixed path
still executes decode and prefill serially around a synchronization. Treat it as
a fairness/latency fix, not the final concurrency engine.

## Current Mia delta (2026-09-01)

Mia's current two-Spark overlay now defaults its E2 fat-expert prefill path and
a 7,168-token prefill ceiling. Atlas has corresponding device-compacted EXL3
large-M execution and a 7,168-row arena. Mia still has two target-path
advantages that matter to decode: FlashInfer SM120 sparse MLA over packed FP8
KV and its production fused EXL3 MoE stack.

The honest median-of-five, 400-token structured gate measures Atlas at 49.60
tok/s with seven drafts. Mia currently publishes 65.1 tok/s for its matching
lab protocol. The short 64-token Atlas canary reports higher numbers because
the TTFT boundary excludes the first speculative burst from elapsed decode
time; do not use it as the cross-engine headline. See
[`NATIVE_TP2_VALIDATION_2026-09-01.md`](../../../results/legacy/2026-09-01-atlas-native-tp2-validation/NATIVE_TP2_VALIDATION_2026-09-01.md).

## Historical EP bootstrap receipts (2026-08-31)

Before the pure-TP2 implementation, the full EXL3 checkpoint completed text,
native tool-call, cancellation/slot-reuse, and C1/C2/C4 endpoint canaries on
EP=2. In the controlled two-request
overlap test, reducing the active-decode prefill slab from 128 to 32 tokens cut
the existing stream's maximum token gap from 2.028 seconds to 0.843 seconds and
raised its effective rate from 2.83 to 4.32 tok/s. The incoming prompt's TTFT
rose from 9.32 to 15.90 seconds, while total wall time stayed near 32.4 seconds.

The final C4 canary completed all requests in arrival order with maximum token
gaps of 0.14-0.75 seconds. TTFT still grew from 7.41 to 40.21 seconds across the
queue: current phase interleaving bounds decode stalls but does not make EP
prefill and decode execute concurrently. Exact commands, per-request numbers,
binary identity, and caveats are in
[`LIVE_VALIDATION_2026-08-31.md`](../../../results/legacy/2026-08-31-atlas-live-validation/LIVE_VALIDATION_2026-08-31.md).

## Reproducible test without a GPU

Download only the official `config.json`, then run:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo run -p atlas-core --example glm53_contract -- /path/to/config.json 32768
```

The important output should be approximately:

```text
layers=45 kda=34 dsa=11 dense=3 moe=42
eos=[154820, 154827, 154829]
kda_recurrent_mib=136.00
kda_conv_mib=9.56
dsa_cache_mib@32768=374.00
persistent_mib_per_sequence=519.58
prefix_cache_kv_only_safe=false
native_execution_implemented=true
end_to_end_validated=false
```

CPU test suites:

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p atlas-core --lib
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo test -p spark-model --no-default-features --lib
```

These prove parsing, topology, byte accounting, factory routing, mixed-dtype
loading contracts, and fail-closed behavior. The GB10 probes below cover the
individual native operators; they do not replace full-model logit/token parity
or endpoint behavior.

### Measured SM121 operator gates (2026-08-31)

`glm53_kda_decode` was compiled with CUDA 13.0 for `sm_121f`, strict warnings,
and `--fmad=false`, then run on one occupied DGX Spark without loading a model.
The deterministic KDA test used three independent heads for 32 sequential
tokens:

```text
heads=3 steps=32 state_max_abs=2.23517418e-08 output_bf16_max_abs=0.000120747834
batch=2 heads=3 slots={3,1} steps=32 state_max_abs=3.7252903e-09 history_max_abs=0 output_bf16_max_ulp=2
tokens=23 heads=3 slots={3,1} conv_max_ulp=0 history_max_abs=0 beta_exact=true norm_max_ulp=0
```

Additional focused receipts:

```text
FlashKDA: tokens=50 heads=4 slots=[3,1] output_exact=true active_state_exact=true unused_state_exact=true
DSA index: pool_chunk_max_abs=0 pool_cpu_max_abs=0 top512_exact=true sparse_mla_max_abs=1.39698386e-09
DSA BF16 projection: absorb_max_abs=6.103515625e-05 expand_max_abs=1.52587890625e-05
EXL3 expert, 8 rows: max_abs=0.0001658872 mean_abs=0.0000268531 finite=true
FP32 router, 128 rows: logits_max_abs=1.2397766e-05 weights_max_abs=8.9406967e-08 ids_exact=true
```

The router probe also found that rounding those same logits to BF16 before
top-k changed at least one selected expert on 27 of 128 rows. That is why the
GLM path treats `moe_router_dtype=float32` as a hard checkpoint contract. These
are correctness receipts, not throughput results. See `bench/glm53/`.

## Two-Spark canary flags

The optimized appliance is fixed pure TP=2. The older EP=2 bootstrap below is
retained only as a historical diagnostic recipe; do not use it as the fast
path. Run `scripts/start-glm53-tp2.sh`, whose actual topology is
`--ep-size 1 --tp-size 2`, max batch 4, DFlash2 eight-row verification,
phase interleaving, and rank-symmetric target graphs on.

Use the same flags on both ranks, changing only `--rank` and making the worker
HTTP port zero. Do not set `ATLAS_HOLO_ALWAYS_MIXED`.

```bash
export ATLAS_EP_PROTOCOL=v2
```

```text
--world-size 2 --ep-size 1 --tp-size 2
--max-seq-len 32768 --max-batch-size 4 --max-num-seqs 4
--max-prefill-tokens 7164 --phase-prefill-slice-tokens 128
--kv-cache-dtype bf16
--scheduling-policy phase-interleave
--phase-decode-steps 1 --phase-prefill-steps 1
--tbt-deadline-ms 500
--swap-space-gb 0
--dflash --draft-model /dflash --dflash-gamma 8 --dflash-window-size 2048
```

The target ranks execute the same TP collective order. Prefix caching and swap
remain disabled because each needs atomic KV + KDA + semantic-index handling,
not just a KV snapshot. DFlash2 rollback is implemented for this fixed path.

Once the endpoint is healthy, run the staggered probe described in
`bench/glm53/README.md`. Compare active-decode slab sizes such as 128, 64, and
32 tokens with identical prompt bytes, forced output lengths, and at least
three post-warmup repeats. Record maximum inter-token gap as well as TTFT and
mean decode rate.

## First honest two-Spark recipe

Start text-only with the routed-expert EXL3 checkpoint, `EP=1`, `TP=2`,
DFlash2 eight-row verification, prefix cache off, HSS off, target CUDA graphs on, and a 32K
context ceiling. The deliberately fixed serving envelope is four live
sequences. Raise it only after expanding the batch workspaces and repeating the
state ledger, collective trace, TTFT distribution, and output-isolation gates.

Interleaving decode and prefill stops one long prompt from monopolizing the
scheduler, but the slab size explicitly trades existing-stream smoothness for
incoming-request TTFT. It does not remove the KDA state tax or accelerate the
target kernels.
The measured 32-token active slab protected streaming much better than 128
tokens at nearly identical total wall time, while making the incoming request
wait longer. The current C2-C4 path does execute one cross-sequence target batch
and identical TP collective order on both ranks.

## Spark validation gates

Completed gates include the SM121 operator probes, full rank-local load, exact
tiny text, native tool parsing, CUDA-graph replay, cancellation/slot reuse, and
C1/C2/C4 endpoint behavior. Remaining gates are:

1. authoritative short-prompt logit parity and agreed greedy-token parity;
2. C8 and fixed-byte short/long workload matrices with p50/p95/p99 TTFT,
   inter-token gaps, useful tokens/s, and power/thermal traces;
3. long cancellation/slot-reuse soak and rank/node failure recovery;
4. profiler-guided EXL3 MoE and sparse-MLA target-kernel optimization;
5. only then, atomic state snapshots followed by prefix reuse.

Do not generalize the measured canary numbers into a production or
cross-engine performance claim before those gates pass.
