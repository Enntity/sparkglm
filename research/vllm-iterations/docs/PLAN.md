# Execution plan

## Phase 0: frozen controls

1. Leave the installed LLooM recipe-v8 service unchanged as the historical
   control.
2. Preserve two public source/configuration controls:
   - Mia `c707598`: current upstream compatibility and kernel baseline;
   - Reederey `b229968`: production scheduler/cache-policy reference.
3. Build one source candidate that contains Mia's required GLM/EXL3 enablement
   and the Reederey changes that are both composable and supported by measured
   receipts. Keep disputed settings as explicit A/B arms.
4. Run all runnable controls through one benchmark harness and archive raw receipts, configuration,
   source/image identity, clocks, memory, temperatures, and rank logs.

## Phase 1: source-equivalent engine

1. Convert Mia's source-exact container patches into ordinary changes against
   the pinned vLLM source tree.
2. Carry Reederey's gate-v3 aging and per-group DFlash cache retention as source
   changes. Keep fine-grained APC optional until a warm/fork workload gate is in
   place; do not carry the unproven router-GEMM experiment.
3. Build a private arm64/CUDA-13 image from source with the same FlashInfer,
   DeepGEMM, EXL3, model, cache, TP2, and DFlash2 contracts.
4. Pass both reference suites plus output, tool, structured-output,
   cancellation, prefix-cache, KDA/DSA state, rollback, and rank-agreement
   checks.
5. Require primary-matrix equivalence before beginning optimization. Equivalence
   means no material regression, not merely that the API responds.

The first configuration matrix deliberately resolves public-reference
disagreements instead of choosing by reputation:

| Setting | Mia arm | Reederey arm | Decision gate |
|---|---:|---:|---|
| max batched tokens | 7,168 | 3,584 | 16K/32K C1-C2 plus both-rank memory floor |
| long-prefill threshold | recipe default | 1,792 | staggered TTFT and decode gap, not solo prefill |
| DFlash draft TP | 2 | 1 | steady decode, acceptance, and per-rank memory |
| DFlash source | latest recipe fetch | `7d74cdd` | output, acceptance, long-context decode |
| mixed-prefill policy | earlier gate | gate v3 + aging | short TTFT, cold-read wall, decode p99 gap |

Mia's E2 fat-expert path is in both arms because Reederey's independent A/B also
found a material cold-prefill win with decode in the noise.

## Phase 2: profile the real workload

Capture synchronized rank traces for medium-C1, medium-C2, large-C1, and
large-C2. Attribute wall time and bytes to routed MoE, sparse indexer, sparse
attention/gather/projection, KDA, TP collectives, DFlash proposal/verification,
cache management, and scheduler idle/wait time. Optimize the largest measured
term, not the most interesting kernel.

## Phase 3: progressive replacement

1. Tune/integrate the E2 fat-expert prefill path for 16K/32K shapes.
2. Replace padded NoPE sparse-MLA/indexer work with an SM121-native fixed-GLM
   path when it beats the current FlashInfer/DeepGEMM combination.
3. Fuse KDA prefill work only where the profile shows avoidable traffic or
   launches.
4. Specialize TP2 communication and overlap for the exact two-Spark topology.
5. Add bounded, rank-symmetric phase interleaving for staggered C1-C2.
6. Improve DFlash2 selection, vocabulary projection, verification, acceptance,
   and graph capture.

Each winning change gets its own commit and receipt. Rejected experiments are
documented and removed from the serving path.

## Phase 4: extraction decision

After the hot path is specialized, profile host/framework overhead again. Build
a thinner C++ or Rust executor only if measured residual vLLM orchestration is a
material ceiling. Reuse the proven kernels and state contracts; do not restart
model implementation from equations a second time.
