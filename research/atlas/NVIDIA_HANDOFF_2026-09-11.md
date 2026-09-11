# Atlas NVIDIA refresh: September 10–11 research handoff

Atlas now has a working NVIDIA NVFP4 bring-up and a substantially faster
prefill candidate. It remains an **experiment**, disabled in SparkGLM's
recommended serving path. The final qualification is incomplete. In particular,
the best completed same-prompt client measurement was r10's 13.067441 seconds
versus the recorded vLLM reference of
10.848938 seconds. That is a remaining 20.45% latency gap, not parity.
The final r11b canary passed at 13.087591 seconds, about 20.63% behind the
reference; it did not establish an endpoint speedup over r10.

The [checksum-bound campaign record](../../results/candidates/2026-09-11-atlas-prefill/RESULT.md)
is authoritative for final measurements, source hashes, settings and the last
attempt's outcome. This narrative explains the sequence and how to resume it.
The [initial bring-up record](../../results/candidates/2026-09-10-atlas-nvidia-bringup/RESULT.md)
remains separate. Neither record promotes Atlas or changes the vLLM default.

## What came from upstream

The new starting point is [Mango-kid/atlas, feat/glm53-dual-spark](https://github.com/Mango-kid/atlas/tree/feat/glm53-dual-spark),
frozen at `90b3584abc71b44b609637092b85d8423d8ff20f`. Its native GLM loader,
NVFP4 arithmetic, KDA/MLA attention, native MTP drafter/verifier, distributed
ownership and serving machinery made this refresh worth attempting. These
are inherited capabilities, not work invented by SparkGLM. Upstream speed
reports motivated the experiment; they are not measurements of our NVIDIA
checkpoint or workload and are not interchangeable with the results below.

The older SparkGLM Atlas archive is separately reconstructed from
[Atlas-Inf/atlas](https://github.com/Atlas-Inf/atlas) at
`bdcccc2ca91eba084aac94a059e3b0f4a5d556dd`, plus archive commit
`775cb3655e29a3735f4f58faa540608f9427bf51`. It supplied the prior GLM semantic
contracts and comparison history. Do not apply that archive patch to the Mango
checkout: the two reconstruction routes have different bases.

The refresh adapted the upstream loader, dispatcher, speculative state and
attention paths. SparkGLM's additional work comprises NVIDIA packed dense
weight validation/loading, the explicit recurrence-normalization opt-out,
unused-predictor omission, channel-preserving stream completion, repaired
one/two-draft policy and long-context index handling, memory accounting,
bounded prefill substitutions, and their regression/oracle tests. cuBLASLt
adapters use NVIDIA library primitives; the transient CUTLASS experiments
reuse the pinned Atlas collective. Neither is a new matrix-multiply algorithm.

All Atlas-derived code and experiments remain **AGPL-3.0-only** under this
directory. Preserve upstream notices. Model weights, converted predictor
tensors, binaries, private launch addresses and raw operational logs are not
distributed. See [licensing](../../docs/LICENSING.md),
[provenance](../../docs/PROVENANCE.md), and `provenance/upstreams.json`.

## Pins and reconstruction

| Component | Frozen identity |
| --- | --- |
| Mango Atlas base | `90b3584abc71b44b609637092b85d8423d8ff20f` |
| Initial adapted engine | `561ffcbf59c12982103984dbbc0060b867b6c315` |
| Final local engine checkpoint | `dd0ffd157fb7c6e96d2f3858c888513c6f142cf6` |
| NVIDIA GLM-5.3-Flash-NVFP4 checkpoint | `423acf37583782c51c142d145aef733d72943d93` |
| CUTLASS build dependency | `cf064d2e6bad2886238ac565b3b49007764f4939` |
| Native compiler/runtime | Rust 1.93.1; CUDA 13.0; NCCL 2.31.2 |

The checkpoint is an external
[NVIDIA artifact](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/423acf37583782c51c142d145aef733d72943d93).
Its publisher and underlying model terms apply independently of the engine.
The native MTP overlay derives from its appended predictor; it is not the
DFlash2 checkpoint used by the vLLM reference.

From a SparkGLM checkout, create a separate source tree:

```sh
SPARKGLM_SOURCE=$(pwd)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Mango-kid/atlas.git atlas-nvidia
cd atlas-nvidia
git checkout 90b3584abc71b44b609637092b85d8423d8ff20f
git apply --check "$SPARKGLM_SOURCE/research/atlas/nvidia-modelopt-refresh.patch"
git apply "$SPARKGLM_SOURCE/research/atlas/nvidia-modelopt-refresh.patch"
git apply --check "$SPARKGLM_SOURCE/research/atlas/nvidia-prefill-mtp2.patch"
git apply "$SPARKGLM_SOURCE/research/atlas/nvidia-prefill-mtp2.patch"
```

The second patch is incremental over the first. Use the result bundle's
checksums rather than assuming a private container tag identifies this tree.
The complete final tree is `e6a4b26058000d1c81b0c87bbf4c2cb0c9127c03`;
the private checkpoint commit is reconstructed by these public patches, not
pushed to the upstream engine remote.
No locally cached image or converted weights are needed to inspect the source.
A real native build needs the pinned CUDA/NCCL/CUTLASS toolchain; CPU-only
checks cannot qualify GPU arithmetic or distributed execution.

## Chronology and measured progress

1. **Initial bring-up.** Direct NVIDIA dense-weight loading and unused-predictor
   omission made ordinary TP2/EP2 serving possible. Chat, retrieval, tool and
   cancellation probes ran, but some answers remained in reasoning or contained
   unrelated reasoning. The first MTP load failed memory preflight. These
   failures remain in the original bring-up record.
2. **Native MTP correctness before speed.** Preparing the native predictor
   overlay removed its BF16 loading peak. Fast four-draft settings could
   repeat JSON/XML or spend the entire budget in reasoning; merely changing
   the target head to BF16 did not establish correctness. A conservative
   repaired path passed bounded canaries. One-draft vLLM experience was not
   treated as evidence that four drafts were safe.
3. **Repaired MTP1, eager then graphs (r4).** Four canaries passed in each arm.
   Eager and graph chat/reasoning and coding outputs were byte-identical in
   the tested cases. C1 full-wall coding throughput rose from the recorded
   non-MTP 13.108 tokens/s to 14.449 eager and 14.814 graph; corresponding
   decode-window figures were 13.431, 14.856 and 15.238. First-draft acceptance
   was 85.5%. Subsequent two-row MLA batching reached 15.58 full-wall and
   16.06 decode-window tokens/s with the same 85.5% acceptance. This was a
   modest gain, not the hoped-for vLLM speed. It did not establish MTP2/C4
   behavior.
4. **MTP2 and actual long-context state (r5).** Work addressed causal sparse
   index construction, private drafter prefix state and rollback/replacement
   around pooling boundaries. It did not replace indexed attention with dense
   attention or merely remove the old context guard. Native oracle failures
   were diagnosed before model testing; unordered top-K output is compared by
   defined selection semantics, not arbitrary output order.
5. **Prefill became the sole performance objective.** C4/context campaigning
   was paused. Small correctness canaries preceded long loads. Profiling
   identified grouped MLA projections, sparse attention, MoE, KDA, HC and
   routing costs. Prefill work retained decode/verification dispatch boundaries.
6. **r6–r7.** BF16 paged projections, batched cuBLASLt MLA projections, sparse
   tensor-core attention with proven K/V reuse, larger chunks, and a separate
   864 MiB/rank cache for the initial three dense FFNs reduced prefill time.
   The dense cache stays separate from decode weights and is reserved before
   final KV budgeting. Exact-shape native tests covered its projections.
7. **r8–r10.** Ordered router blocking and HC prefill work were retained. KDA
   projections gained a temporary FP8 cuBLASLt route whose timings include
   conversion. Larger chunks exposed startup over-reservation: global SSM
   heads were charged before TP-local topology. The repaired accounting applies
   topology once and still counts real arena rows, live/MTP dummy slots and
   private speculative state. r10 then completed the selected long prompt.
8. **r11 streaming tail.** Source tracing showed that the scheduler already
   enqueues the sampled prefill token immediately. The content sanitizer was
   retaining eleven bytes even for harmless prose. Its replacement retains
   only a suffix that is a strict prefix of a known marker, preserving complete
   marker, suppression, UTF-8 and channel rules. CPU tests passed; the first
   r11 model load was refused at 7,954 target KV blocks against the required
   8,196. A guarded retry at GPU utilization 0.914 loaded both ranks and passed
   four short canaries. Its final field-guide canary also passed: 15,807 input
   tokens, eight output tokens and no reasoning. Client first-content was
   13.087591 seconds, server first-token 13.054014 seconds and full wall time
   13.852895 seconds. A failed initial load is not a timing result.

The early synthetic retrieval fixture had **15,762 actual input tokens**:

| Stage | Server first-token seconds | Client first-content seconds |
| --- | ---: | ---: |
| r5 initial | 41.481070 | 41.849316 |
| r5 BF16 paged projections | 29.085716 | 29.437675 |
| r6 attention, chunk 1024 | 18.943122 | 19.305593 |
| r6 attention, chunk 2048 | 17.061549 | 17.415016 |
| r7 dense cache | 14.556071 | 14.920482 |
| r10, chunk 4096 | 13.058678 | 13.430328 |

The later **field-guide fixture** used the frozen first prompt from the
staggered C4 workload, run alone with eight output tokens. Its prompt SHA-256
is `d707531169c27ae0b2fb19c9db120947738378f30793ed03f8611c3472d41bd6`.
r10 measured 15,807 input tokens, 12.708584 seconds server first-token and
13.067441 seconds client first-content; its r7 client control was 14.890740
seconds. The recorded vLLM first-client-content
reference was 10.848938 seconds. These are configuration comparisons, not an
isolated kernel A/B: engines, speculation and internal precision differ.
The two prompt fixtures must not be spliced into a single matched speedup.
Server TTFT and client content TTFT are different metrics. The approximately
0.36-second r10 difference motivated r11; not every millisecond was proven to
come from sanitizer buffering.

In r11b the client/server first-token difference fell from 358.857 ms to
33.577 ms, and the first visible content began with the first token. However,
server first-token time increased by 345.431 ms in that run, offsetting the
earlier emission. The resulting client time was effectively unchanged in this
single comparison. This establishes the intended streaming behavior in the
canary, not a statistically established endpoint performance improvement.
No further GPU experiments were run after that final canary.

## Rejected or incomplete routes

| Route | Disposition and reason |
| --- | --- |
| General FP4 prefill activation path | Rejected after a short arithmetic answer regressed; no long-prefill win accepted |
| Aggressive four-draft native MTP profile | Failed bounded answer-quality checks; retained only as diagnostic history |
| Temporary FP8 conversion plus the older custom GEMM | Output checks passed but conversion-inclusive time did not improve |
| MoE M128, K128 and alternative B tiling | Standalone alternatives did not justify replacing the selected M64 path |
| HC post vectorization | Small gain only; baseline already reuses all four HC residual inputs per hidden coordinate |
| HC TF32 heuristic search | About 1.04× in the tested case; rejected against the 1.5× continuation gate |
| Sparse QK work redistributed across eight warps | Numerics passed but 15.984480→16.436096 ms was 0.973×; rejected |
| Transient-SFB grouped CUTLASS | Numerical gates passed; preparation-inclusive projection speed was 0.803× for ping-pong and 0.852× for cooperative K256; rejected, with no full-FFN claim |
| 8K prefill chunks | Retained fallback, not selected: roughly 7% kernel-only gain estimated at about 1.4% overall, with roughly 2.4 GiB extra allocation; no completed serving qualification |
| r11 first load | Correctly refused the physical KV floor; no OOM or performance conclusion follows |

## Resume with bounded iteration cost

The [portable resume assets](nvidia-prefill/README.md) provide the pinned build
recipe, external MTP-overlay preparation steps and a render-only two-rank
Docker profile matched to r11b. Its clean Docker build has not been rerun;
the recorded performance uses the cached native build. Execute rendered
commands manually, worker first, only after a deliberate service handoff.

Use the campaign record's last **completed** canary profile as the control.
Keep every source tree, image and checkpoint identity fixed before changing a
single option. Preserve the four-GiB free-memory guard, actual post-allocation
KV checks and the 8,196-block floor required by the four-slot 32K capacity
contract. Reserving that capacity does not qualify staggered C4 service.
Do not disable safeguards to make an 8K chunk or another weight cache fit.

The retained prefill switches include `ATLAS_GLM_PAGED_PREFILL_BF16_GEMM=1`,
`ATLAS_GLM_PAGED_PREFILL_MLA_GEMM=1`, `ATLAS_GLM_SPARSE_PREFILL_TC=1`,
`ATLAS_GLM_SPARSE_PREFILL_KV_REUSE=1`, `ATLAS_GLM_DENSE_PREFILL_BF16=1`,
`ATLAS_GLM_ROUTER_PREFILL_BN32=1`, `ATLAS_GLM_HC_PREFILL_VEC=1`, and
`ATLAS_GLM_KDA_PREFILL_LT_FP8=1`. These are not a complete launch recipe:
MTP repair, distributed ownership, precision, chunk and memory settings must
come together from the exact recorded profile. Keep `ATLAS_FP4_PREFILL` absent.
Keep the BF16 target head and the documented explicit
`ATLAS_GLM_SSM_NORMALIZE=0` semantic choice. The latter is not proof of parity
with vLLM or a general correction to upstream Atlas.

For a local GPU-free protocol regression check in the reconstructed tree:

```sh
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
  cargo test --locked --no-default-features --features metal \
  -p spark-server --bin spark sanitizer_prefix
```

This command is the tested macOS CPU lane, not a Linux/CUDA build recipe.
Before another full-model load, run the affected native operator/policy tests
with a separately available GPU and explicit allocation bounds. Then run a
short correctness canary, the exact frozen field-guide request, and compare
both content and reasoning with the control. Measure first client content
from SSE events, retaining server TTFT separately. Require the normal repeated
workload discipline before describing a result as a durable performance win.

Only after prefill reaches its target should work resume on MTP2 acceptance,
32K rollback correctness, staggered C4 fairness/throughput, cancellation,
memory pressure and endurance. Broad semantic qualification and the repository's
tinyGLM integration gate remain incomplete. Follow
[METHODOLOGY.md](../../docs/METHODOLOGY.md); kernel speed or one successful
request does not promote this engine.

The next technical priority is a fresh whole-prefill critical-path profile of
the frozen control: separate target MoE, KDA recurrence and sparse attention
from communication and host gaps, then choose the component with the largest
measured wall-time contribution. Require a substantial, preparation-inclusive
native win before spending another model load. The rejected tile/heuristic
families above are not a queue of tweaks to rerun. An 8K chunk remains a
bounded fallback if its actual allocation headroom and end-to-end benefit can
be demonstrated; it is not the selected next optimization.
