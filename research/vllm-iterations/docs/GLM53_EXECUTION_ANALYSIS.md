# GLM-5.3 execution analysis on two GB10s

Date: 2026-09-02

This is the durable execution model and optimization map for the SparkGLM
appliance. It distinguishes unavoidable GLM-5.3 work from implementation cost,
and model-specific optimization from exact GB10/TP2 specialization.

## Verified appliance shape

The live two-Spark service was inspected on both ranks before this analysis.
Its target configuration is:

- 45 transformer layers: 34 KDA linear-attention layers and 11 sparse NoPE-MLA
  layers;
- dense MLPs in layers 0-2 and routed MoE in layers 3-44;
- 288 routed experts, top 8 experts per token, 2,048 global expert intermediate
  width;
- four mHC residual streams;
- sparse index top-2,048 with four-token K pools;
- TP=2, maximum four sequences and a 2,048-token engine-step budget;
- packed FP8 MLA KV cache;
- DFlash2 with seven speculative tokens and a TP1 drafter;
- exclusive phase interleave: eight decode-only steps followed by one prefill-only
  slab of at most 2,048 tokens while decode and prefill contend.

The live engine selected `FLASHINFER_MLA_SPARSE_SM120`, the fused EXL3 E2
kernel tier on both ranks, and PyNCCL for cross-node TP collectives. Custom
all-reduce was unavailable because the two-node group has no MNNVL multicast.
Sequence parallelism, GEMM/communication fusion, and all-reduce/RMSNorm fusion
were disabled.

## Simple execution model

TP2 does not assign half the layers to each Spark. Both ranks execute every
layer on the same token batch. They split attention/KDA heads and inner weight
dimensions, then combine partial output at row-parallel boundaries.

Decode advances a few rows through already-built state. Prefill manufactures
that state for thousands of rows while the sparse selector repeatedly searches
an increasingly large history. This shape difference is the main reason decode
is much closer to an efficient implementation than long-context prefill.

## Fundamental work by operation

### mHC residual mixing and normalization

GLM maintains four residual streams. Each layer mixes them, normalizes the
selected input, runs attention/KDA and its MLP, and mixes the result back.

Current strengths:

- adjacent mHC post/pre operations are fused across layer boundaries;
- RMS normalization is fused into those operations;
- packed projection outputs avoid padded RMSNorm reads.

The packed-output change measured +9.7% effective prefill at medium C1 and
+2.7% at large C1. This is model-specific and useful, but its declining share
at longer context shows it is not the principal 32K bottleneck.

Classification: GLM-specific and reasonably fused; not especially GB10-specific.

### KDA: 34 layers

KDA projects Q, K, V and gate terms, applies a four-token causal convolution,
and advances a recurrent 128x128 state per head.

Current strengths:

- six input projections are merged into one GEMM;
- Q/K/V use one merged short-convolution call;
- TP2 divides 64 KDA heads into 32 per rank;
- decode computes gate and beta transforms inside the recurrent kernel;
- plain and speculative decode can write recurrent output directly into the
  layer output buffer.

Prefill still invokes a multi-stage Flash Linear Attention pipeline: Q/K
normalization, gate cumulative sum, chunk QK product, triangular solve,
intermediate reconstruction, recurrent-state advance, output formation, and
state scatter. These are separate kernels and global-memory intermediates. The
KDA body is also an eager break inside the otherwise captured graph because its
state control is host-dependent.

Classification:

- KDA decode: strongly optimized, model-specific, but not an exact GB10 kernel;
- KDA prefill: correct and partly fused, with meaningful remaining launch and
  memory-traffic opportunity.

### Sparse index selection: 11 layers

The indexer currently performs:

1. index Q/K and learned head-weight projections;
2. K normalization;
3. Hadamard rotation plus FP8 Q quantization;
4. softmax-weighted four-token K-pool compression and cache insertion;
5. paged historical-K gather into a contiguous workspace;
6. every-query versus every-eligible-pool score calculation;
7. materialization of the complete FP32 score matrix;
8. a separate exact top-k selecting 512 pools;
9. expansion of those pools into up to 2,048 token indices;
10. request-relative to physical cache-address conversion.

Several front-end and tail-cache operations are already fused. The expensive
interior remains a gather -> score -> materialize -> top-k -> expand -> address
conversion pipeline.

At a late 32K slab, four-token pooling leaves about 8,192 candidate pools. A
2,048-row slab can therefore materialize approximately 2,048 x 8,192 FP32
scores: about 16.8 million values or 64 MiB per sparse layer per rank before
top-k. At 16K the corresponding late-slab matrix is about 32 MiB. Exact sizes
vary by each row's causal bound.

This exposes an important architecture/implementation boundary:

- fundamental: exact GLM selection must compare each query with the eligible
  compressed history unless the selection algorithm itself changes;
- avoidable: it does not need to write the entire score matrix to HBM, read it
  again in a separate top-k kernel, expand logical indices separately, and then
  translate those indices in another pass.

For contexts at or below 2,048 tokens, the engine skips selection because all
causal tokens fit inside top-k. A 16K prompt therefore has roughly one cheap
slab followed by seven genuinely sparse slabs. A 32K prompt has one cheap slab
followed by fifteen increasingly expensive sparse slabs. The selector's total
exact work consequently grows approximately quadratically with prompt length,
divided by the four-token pool compression, even though the final value
attention is sparse.

The indexer is replicated across TP ranks. A tested query-row sharding port
removed duplicate score/top-k work but exchanged roughly 16 MiB of selected
indices per full slab per sparse layer over CX-7. It regressed C1 prefill by
4.4-6.2% and was reverted. Replication is not theoretically optimal; the
naive full-result all-gather is worse on this topology.

Classification: heavily adapted to GLM semantics, but the least appliance-native
part of the long-prefill path.

### Sparse MLA attention

After selection, the engine calls the Blackwell-family FlashInfer sparse-MLA
backend with packed FP8 MLA cache.

Current strengths:

- native CUDA implementation for compute-capability family 12;
- exact top-2,048 contract and FP8 DS-MLA cache support;
- TP2 divides the 64 attention heads into 32 local heads.

Compatibility costs:

- GLM has a 512-wide NoPE latent geometry, but this backend pads 64 zero RoPE
  dimensions to use an existing 576-wide GLM-NSA geometry;
- `supports_dense_mha_prefill` is false;
- both live ranks report that prefill uses only the top-k MQA path;
- the implementation invokes FlashInfer's TRT-LLM batch-decode MLA API for all
  rows, including a 2,048-row prefill slab;
- pool expansion and logical-to-physical conversion occur before the attention
  kernel rather than inside a GLM prefill pipeline.

This does not prove dense attention would be faster at 16K/32K. It proves the
backend has no prefill-specialized execution path.

Classification: a good Blackwell-family kernel, but compatibility-shaped for
GLM NoPE prefill rather than exact GB10/GLM specialization.

### Routed MoE and EXL3: 42 layers

Each token is routed to 8 of 288 experts. One 2,048-token slab creates 16,384
expert-token assignments per MoE layer, or about 688,000 across 42 layers,
batched by expert. Each rank holds all 288 expert identities and one half of
each expert's intermediate dimension: 1,024 per rank versus 2,048 globally.

Current strengths:

- EXL3 weights stay packed instead of expanding into persistent BF16 experts;
- thin/decode shapes use one fused `exl3_moe` launch per layer;
- the E2 direct gate/up kernel and weighted down/scatter kernel are built into
  the SM121 extension;
- dispatch adapts according to expert occupancy;
- both live ranks report the E2 kernel tier, correct TP2 dimensions, and all
  required extension symbols.

Remaining costs:

- router GEMM, top-k, sorting and count construction use generic vLLM fused-MoE
  infrastructure;
- the large-row fat-expert path stages expert counts to the CPU and synchronizes
  before dispatching overflow experts;
- each rank duplicates routing and keeps all expert identities;
- the partial expert output requires a TP reduction.

Classification: strongly GLM/EXL3/SM121/TP2-specialized inside the expert GEMMs;
generic around routing and communication. EXL3 is not the main cause of the
long-context prefill collapse.

### TP2 communication

Attention/KDA output projections and dense/MoE outputs are row-parallel, so a
target forward has roughly two synchronization boundaries per layer--on the
order of 90 collectives.

- Prefill collectives carry large 2,048x4,096 activation tensors and are
  bandwidth-sensitive.
- Decode collectives carry small Bx8x4,096 tensors and are primarily latency
  and synchronization sensitive.

The current service uses generic PyNCCL over CX-7. It has no exact two-rank
communication schedule, collective/GEMM overlap, fused all-reduce-plus-norm,
or sequence-parallel GLM plan.

Classification: correct conventional TP, not a native two-Spark execution plan.

### DFlash2 decode

The TP1 Qwen3-derived DFlash2 drafter proposes seven tokens. The TP2 target then
verifies up to eight rows per request through the complete 45-layer model.

At 32K, one sparse layer scores approximately 2,048 x 8,192 values for a full
prefill slab but only 8 x 8,192 values for a single-request target verification.
The 256x query-row difference is why decode is much better behaved.

Decode still contains compatibility machinery:

- SM121 uses flattened speculative rows because the indexer reports no native
  variable-length paged-MQA support;
- request block tables and sequence lengths are expanded for that layout;
- the kpool persistent CUDA top-k branch is disabled, so the generic per-row
  top-k fallback runs;
- candidate vocabulary projection, top-k and rejection sampling remain separate
  general-purpose operations.

Classification: effective generic speculation with GLM correctness adapters;
not a fused GLM/GB10 decode engine.

### Scheduler and hybrid cache

The accepted q8/p2048 scheduler is exact to the appliance. Under contention it
alternates eight decode-only engine steps with one prefill-only slab. It never
mixes a long prefill and decode rows in the same target forward.

This measured:

- medium staggered C2: B TTFT -40.5%, aggregate delivery +36.6%, wall -23.0%;
- large staggered C2: B TTFT -13.6%, aggregate delivery +20.5%, wall -12.2%.

This is a fairness and throughput workaround around expensive prefill, not a
prefill speedup. The prefill slab still pauses active decode.

The cache layer is generic vLLM hybrid-state machinery with GLM adapters for
KDA recurrent/conv state, sparse MLA pages, K pools and tails, DFlash sliding
state, speculative rollback, and prefix retention. Its complexity is required
for correctness, but it is not a unified GLM-native state layout.

## Specialization scorecard

| Area | Current implementation | GLM-specific | GB10/TP2-specific |
| --- | --- | ---: | ---: |
| Model/layer contract | dedicated GLM model | high | neutral |
| mHC mixing/norm | fused GLM operations | high | medium |
| KDA decode | fused recurrent/conv/gate path | high | medium-high |
| KDA prefill | multi-stage Flash Linear Attention | medium-high | medium-low |
| Sparse projection/compression | several GLM fusions | high | medium |
| Sparse prefill score/top-k | gather + full logits + separate top-k | medium | low |
| Sparse MLA attention | FlashInfer SM12x + FP8 cache | medium-high | medium |
| NoPE geometry | zero-padded to 576-wide kernel | low | low |
| EXL3 expert GEMMs | SM121 E2 direct/scatter kernels | high | high |
| MoE routing/sorting | generic vLLM fused-MoE runner | medium | medium |
| TP collectives | generic PyNCCL over CX-7 | low | low |
| DFlash2 | generic speculative stack with GLM adapters | medium | medium |
| Scheduler | exclusive q8/p2048 phase interleave | high | high |
| Hybrid state/cache | generic framework plus GLM adapters | medium-high | medium-low |

## Optimization order

Every change must preserve exact selection, request isolation, output quality,
rank agreement, and speculative acceptance. A faster incorrect result is a
failed result.

The synchronized current-build profile in
`benchmarks/sparkglm/results/2026-09-03-current-long-prefill-profile.md`
supersedes the pre-E2 ordering below. On two genuine long-prefill target
forwards, the complete routed-MoE boundary consumed about 48-49% of CUDA time,
TP all-reduce 10-12%, mHC about 9%, and sparse MLA about 9%. The complete
sparse indexer consumed only about 0.55%; its scorer and top-k kernels together
were about 0.21%.

### 1. Exact SM121 EXL3 expert compute

Keep the accepted E2 routing and fat-expert dispatch. Optimize the compute
inside the engaged kernels:

1. tune the direct trellis fat GEMM geometry against the measured live row
   distribution, not a uniform or one-expert shape;
2. replace the inherited Ampere-style `mma.sync` inner loop with a genuinely
   SM121-native implementation where it wins;
3. reduce the fat path's gather, input Hadamard, SwiGLU/Hadamard and scatter
   traffic without changing the K4 MCG arithmetic contract;
4. evaluate a persistent hot-expert BF16 cache only after measuring per-layer
   expert stability and including its memory cost.

The paired gate/up and down/scatter direct kernels alone consumed about 25% of
captured CUDA time. The complete MoE boundary is the only current term large
enough for a single kernel project to produce a double-digit endpoint gain.

### 2. Exact two-rank CX-7 communication schedule

Fuse or overlap row-parallel reductions with residual/norm and subsequent
work. Treat prefill bandwidth and decode latency as different schedules. Do
not assume a same-node custom-all-reduce result applies to the cross-node
appliance. The current profile measured TP all-reduce at 10-12% of CUDA time,
with a material rank asymmetry.

### 3. KDA prefill fusion

Reduce global-memory intermediates and launches across gate cumulative sum,
chunk QK/triangular solve, recurrent-state advance, output, and final-state
scatter. Preserve FP32 state semantics. Existing BF16 projection GEMMs were
already close to their weight-read bandwidth; focus on fewer bytes and fused
state movement rather than another GEMM heuristic search.

### 4. Exact sparse-MLA prefill pipeline

Target the fixed appliance shape: 2,048 query rows, kpool=4, exact top-512 pools
expanded to top-2,048 tokens, 32 local attention heads, NoPE width 512, TP2.

Progressively remove these boundaries:

1. avoid the complete FP32 score matrix by maintaining exact top-512 candidates
   while score tiles are resident;
2. operate directly on paged K-pool cache where that beats a contiguous gather;
3. emit physical indices directly;
4. fuse pool expansion and mandatory tail inclusion;
5. use a native 512-wide NoPE attention geometry;
6. consume selection directly in a prefill-shaped sparse-MLA kernel;
7. evaluate intentional duplicate selection against a compact, overlapped TP2
   exchange--never repeat the losing full-index all-gather.

The current profile changes the emphasis. Sparse MLA attention is material at
about 9% of CUDA time, but the complete score/top-k indexer is only about
0.55%. Optimize the 512-wide NoPE attention geometry and its data movement
before attempting another online selector. Implement every eliminated
boundary with a correctness oracle and endpoint receipt.

### 5. Decode selector and DFlash path

Replace the disabled generic top-k fallback with a GB10-safe persistent or
online selector, reduce flattened metadata work, and then optimize DFlash
vocabulary projection/candidate selection/verification. Fixed K=7 remains the
control; a smaller K already lost end to end.

### 6. Host/runtime thinning

Only replace vLLM orchestration after the GPU and TP hot paths above are
specialized and a new profile proves host/framework overhead is a material
ceiling. Reuse the established correctness, scheduling, cache, cancellation,
tool and structured-output contracts.

## Evidence boundary

Measured evidence currently establishes:

- the live configuration and backend choices above;
- packed RMSNorm gains;
- E2 kernel engagement;
- phase-interleave gains;
- the TP row-sharding regression;
- catastrophic mixed-prefill/decode interference;
- earlier Atlas first-chunk attribution to sparse DSA/NoPE-MLA.

The source makes the avoidable stage boundaries explicit. A synchronized
Nsight Systems/Compute trace of the current accepted engine is still required
to assign exact current wall-time percentages to sparse selection, sparse
attention, KDA, MoE and collectives. Until then, expected endpoint gains from
new kernels are hypotheses, not measured claims.

## Optimization execution log

### Accepted: native FP16 sparse scores and selector

Commit `50d5b77fdb` keeps the materialized indexer score matrix in FP16 and
selects it with native FP16 top-k. On the paired five-second-stagger workload,
medium C2 wall time improved 4.85% and large C2 improved 3.18%. The 2,048-row
selector kernel itself improved 37.0% at 4,096 pool columns and 43.5% at 8,192
pool columns. The full receipt is
`benchmarks/sparkglm/results/2026-09-02-fp16-sparse-selector.md`.

This halves the score tensor but does not remove it. Scorer-plus-selector
fusion remains optimization 1's principal target.

### Rejected as a no-op: remove 16-to-32 indexer-head padding

Source comments suggested that GLM-5.3 used 16 indexer heads and was padded to
32 for DeepGEMM. An isolated SM121 test showed native 16-head scoring was
bitwise equal and 1.20-1.55x faster than zero-padded 32-head scoring. The actual
deployed EXL3 checkpoint config was then checked directly: it declares
`index_n_heads=32`, `index_head_dim=128`, `index_kpool=4`, and
`index_topk=2048`. The padding branch is never entered on this appliance.

A whole-engine candidate with an intentionally strict 16-head assertion failed
during its profile run, proving that the tempting kernel result addressed the
wrong shape. The source change was reverted without an optimization commit.

### Rejected: separate DFlash lookahead buffer from scheduled work

A C4 candidate raised `max_num_batched_tokens` from 7,168 to 7,196 while
holding `max_num_scheduled_tokens` at 7,168. This exactly supplied four requests
with seven DFlash2 lookahead slots each, but changed the worker and E2 maximum
shape to recover at most 0.39% more input work per full slab. On the identical
four-stream 33K gate it regressed delivered throughput 2.67%, aggregate decode
3.70%, and wall time 2.74%. It was reverted. The receipt is
`benchmarks/sparkglm/results/2026-09-02-spec-slot-budget-rejected.md`.
