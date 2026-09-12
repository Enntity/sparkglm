# Atlas day two: faster, still behind the retained vLLM appliance

**Experimental, G0 publication record only.** This bundle preserves bounded
operator tests, model canaries and individual C4 observations. It does not
complete the full G1–G5 methodology, promote Atlas, or change the vLLM default.
All published measurements and transformed receipts are checksum-bound by
[qualification.json](qualification.json). [source-map.json](source-map.json)
maps original private receipt hashes to selected public fields; exported files
are not falsely labeled byte-identical private originals.

The latest full C4 observation completes 1600 outputs in **137.129172 seconds**,
versus **205.467964 seconds** at the first day-two C4 checkpoint and
**90.500192 seconds** for retained vLLM. Atlas improved, but still takes
**1.515236×** the reference wall time. Different continuations and speculative
acceptance prevent attributing the whole change to individual kernels.

## Prefill: one field prompt, eight output tokens

These are client time-to-first-visible-content observations for the exact
15,807-token SYSTEMS field prompt, not four concurrent completions. Server-only
TTFT excludes some delivery costs. Raw usage, visible content and both timing
fields are in [prefill-summary.json](prefill-summary.json) and its linked receipts.

| Configuration | Client TTFT seconds |
| --- | ---: |
| Earlier r11b control | 13.087591 |
| r14 two-interface transport | 12.775480 |
| r15 FlashKDA prefill | 11.730531 |
| r16 control, same prewarm comparison | 12.204177 |
| r16 HC prewarm | 11.558193 |
| r17 native sparse continued prefill | 11.238014 |
| r18 BF16 TC decode | 11.276282 |
| r19b grouped C3 MoE | 11.665304 |
| r20 batched vocabulary head | 11.295292 |
| r21 batched MLA output | 11.277983 |
| r23 S8 split decode | 11.259161 |
| Retained vLLM reference | 10.848938 |

The r16 control had unexplained TF32 spikes; its whole difference is not a
clean estimate of prewarming benefit. Later decode changes should not be
credited with prefill gains from this small, noisy table. This series did not
alternate three complete appliance pairs or establish a confidence interval.

## Full staggered C4

Four exact field-guide prompts, arrival offsets 0/1/2/3 seconds, actual prompt
tokens `[15807,15810,15810,15809]`, 400 outputs each. Every per-request record,
visible stream event, usage field and prompt hash is preserved in the selected
`raw/c4-*.json` files. [C4 summary](c4-summary.json) includes their links,
per-request TTFTs and source identifiers. The reference is the existing
[NVIDIA DFlash2 bundle](../2026-09-09-nvidia-redhat-video/RESULT.md).

| Candidate | Total wall seconds | Outputs / total wall second | Conditional draft-position acceptance |
| --- | ---: | ---: | ---: |
| r17 | 205.467964 | 7.787102 | 61.2273% |
| r18 | 169.723908 | 9.427075 | 61.1421% |
| r19b | 159.911387 | 10.005541 | 55.3571% |
| r20 | 151.992779 | 10.526816 | 53.8411% |
| r21 | 139.824123 | 11.442947 | 61.3827% |
| r23 | 137.129172 | 11.667831 | 55.0792% |
| Retained vLLM | 90.500192 | 17.679521 | Not comparable from retained suite counters |

This is an appliance comparison: Atlas uses native MTP2, BF16 KV, FP32 SSM,
32K/four-owner capacity and its documented tuned kernels; vLLM uses DFlash2
MXFP8 TP2 depth 7, FP8 KV and a different tuned configuration. It does not
isolate one implementation change. Each Atlas row is one retained run.

The `raw/acceptance-*.json` exports retain before/after outcome counters,
accepted usage and Done-derived reconstruction. Bootstrap and terminal verifies
are accounted for separately; output clipping can include accepted positions
beyond visible output. Hardcoded zero rejected-token usage is not a denominator.
r23 first-position acceptance is **519/758 = 68.4697%**, distinct from
**835/1516 = 55.0792%** across both proposed positions. Its verifies increased
from 716 at r21 to 758; ROBOTICS contributed 41 of the 42 extra verifies.
The older vLLM native-MTP1 rate 88.7–92.5% used isolation prompts, not these field
guides. DFlash7's 4002/17895 suite counters are also not a matched first-position
rate. [Comparison limits](raw/mtp-comparability.json).

## Chronology and decisions

1. **Review r12/r13 before further optimization.** Sparse worklist changes were
   rejected at reviewed 19.041/20.062s against 13.088s. These numbers are supported
   by a retained independent live/source review; primary per-request JSON is
   absent from this export. GPU tensor correctness was unqualified. Corrected
   NCCL attribution found 551.755ms overlapping other kernels within 1892.890ms
   NCCL union. Neither that union nor D2H API wait is recoverable transfer time;
   short text equality was not numerical proof. [Review evidence](raw/r12-r13-review-summary.json).
2. **r14 transport.** A corrected native standalone compared SendRecv+BF16 add
   with AllReduce. Two-interface SendRecv had lower warm medians and was retained
   for the model sample above. Native AllReduce remained a standalone result;
   noisy tail latency prevented turning its median into an engine claim.
   **N64 MoE was rejected** despite fewer registers and higher theoretical
   occupancy. [Transport and N64](raw/transport-n64-summary.json).
3. **r15 FlashKDA.** A first adapter normalized Q/K twice; tiny-value oracles
   caught it. Corrected raw packing, state transposes and continuation checks
   passed before strict prefill-only integration. Preparation-inclusive 4096-row
   operator medians were 11.668128→3.268480ms. Four short model checks, the field
   probe and a 16,762-token recall passed. This did not qualify full C4 for r15.
   [Operator samples](raw/operator-flash-kda.json).
4. **Reject two fusion screens.** Fused gate/up/SiLU/FP4 passed arithmetic but
   missed 1.2×; HC post/raw projection/RMS fusion missed 1.8×. Their original
   fixed gates stand. A function-local launch-bounds follow-up reduced registers
   but introduced 232-byte stack/spill reports, so it stopped before GPU execution.
5. **r16 HC startup prewarm.** Checked dead-arena shapes and code loading before
   KV sizing removed a first-use HC module stall. Same-image control/warm results
   are retained separately; no whole-model bit-identity claim follows.
6. **r17 native sparse prefill.** Initial 2048-ID screening could not handle the
   required 2051-ID contract. Exact main/tail splitting plus a Torch merge missed
   its 1.3× gate; a reviewed fused merge passed. The unified allocation-free C ABI
   then passed four production shapes and 21 rejections, 32.362530→22.559839ms
   inclusive. Native initialization moved before KV sizing. The first full C4
   exposed much slower serial target verification. [Native records](raw/native-unified-check.json),
   [failed Torch merge](raw/native-paired-tail-result.json),
   [fused merge](raw/native-paired-tail-fused-result.json).
7. **r18 BF16 TC sparse decode.** A 14-fixture standalone compared unchanged
   production BF16 kernels at 2.2153× before allowing a narrow repaired-K3 opt-in. Actual
   layer tests preserved canonical selected IDs/cache, dense fallback and
   verification state. [Standalone record](raw/operator-bf16-sparse-decode.json).
   Two initial fixture/build failures are retained separately from the
   [successful build](raw/build-r18.json). The first C4 attempt had duplicate `/v1`, four HTTP 404s
   and no inference; it remains a harness failure, separate from the valid run.
8. **r19/r19b grouped C3.** Same r18 source/binary, one compute flag. Initial
   r19 failed post-load memory reserve before readiness, so it is not a model
   result or proof of extra C3 allocation. The unchanged retry loaded, proved
   grouped dispatch and completed the C4 row above.
9. **r20 vocabulary head.** Existing M3 batchm beat the M3 GEMM screen at 2.7117×.
   It matched three scalar GEMVs bitwise; the generic GEMM had small BF16 rounding
   differences. Focused integration checks and bounded model probes preceded C4.
   [Head screen](raw/operator-head-batchm.json).
10. **r21 MLA output.** Within-owner M3 batchm passed 2.8265× versus scalar.
    Broader M12's incremental 1.2916× missed the separate 1.3× gate. Only M3 was
    integrated. **r22 diagnostic** then compared all 12,288 BF16 O elements in
    each of 11 layers on each rank, with finite bit identity. It does not prove
    whole-model determinism and its synchronization excludes timing comparison.
    [MLA screen](raw/operator-mla-o-batching.json), [r22 summary](raw/mla-o-diagnostic-summary.json).
11. **r23 S8 split decode.** All 105 numerical records and five fixed inclusive
    S8 timing cases passed; S16 was exploratory. Source/actual-layer tests,
    bounded model probes and profile structure preceded C4. Target sparse
    attention fell 12.94997→2.05912ms. With changed acceptance, the observed C4
    wall reduction was 1.9274% versus r21. [Split screen](raw/operator-split-decode.json),
    [profiles](raw/profile-r23-split.json).
12. **Test ownership batching without changing the engine.** A generic copy
    helper passed only after independent repair. The fixed balanced/skewed M12
    screen failed its all-cases gate. **r24 route capture** therefore instrumented
    eight actual four-ready rounds, 32 owner observations and 1344 layer records. All completed;
    count ratios describe opportunity, not speed. Its build history retains a
    stale-mtime/zero-test rejection and a finished-fixture test failure before
    the corrected native gate passed. [Diagnostic](raw/route-diagnostic-summary.json),
    [final build](raw/build-r24-route.json).
13. **Actual-route replay and remaining routes.** Frozen rounds 1 and 8, all 42 MoE
    layers, synthetic activations/weights and real private captured routes passed
    the 1.3× inclusive routed-stage screen at 1.4561× / 1.4435×. No cohort integration
    followed. Compact-down and matrix-TP screens lost. Three strict-order router
    arms all missed their fixed 2× gate. Reviewed generic profile helpers reproduced
    the exact local trace totals. [Replay](raw/operator-moe-route-replay.json),
    [router](raw/operator-ordered-router-workers.json),
    [independent cross-check](raw/profile-r23-independent-crosscheck.json).

## Standalone outcomes, including rejections

All ratios below compare the stated warm operator/pipeline, not C4. The linked
JSON retains actual samples where available, original fixed gates, precision,
guards and skipped cases. A source-review or compiler failure is not a GPU pass.

| Screen | Observed ratio / gate | Decision and receipt |
| --- | --- | --- |
| N64 MoE | 0.8472 / 1.15 | [Rejected](raw/transport-n64-summary.json); skew/tail skipped |
| Fused MoE epilogue | 1.1383 / 1.2 | [Rejected](raw/operator-moe-fused-epilogue.json) |
| Fused MoE lb3 | 232-byte spills / zero | [Compiler rejection](raw/lb3-compiler-summary.json); no GPU |
| HC fusion | 1.2384 / 1.8 | [Rejected](raw/operator-hc-fusion.json) |
| Native sparse Torch tail merge | 1.0337 / 1.3 | [Rejected](raw/native-paired-tail-result.json) |
| Native fused tail merge | 1.3275 / 1.3 | [Scoped pass](raw/native-paired-tail-fused-result.json) |
| Full native sparse wrapper | 1.4345 / 1.3 | [Scoped pass](raw/native-unified-check.json) |
| Head batchm | 2.7117 / 1.3 | [Scoped pass](raw/operator-head-batchm.json), r20 integrated |
| MLA O M3 | 2.8265 / 1.3 | [Scoped pass](raw/operator-mla-o-batching.json), r21 integrated |
| MLA O M12 incremental | 1.2916 / 1.3 | [Rejected for cohort](raw/operator-mla-o-batching.json) |
| BF16 TC sparse decode | 2.2153 / 1.3 | [Scoped pass](raw/operator-bf16-sparse-decode.json), r18 integrated |
| Matrix-TP MoE | 1.0027 / 1.3 | [Rejected](raw/operator-moe-tp-shape.json); later cases skipped |
| Compact MoE down | 0.9460 / 1.3 | [Rejected](raw/operator-moe-down-compact.json); later cases skipped |
| Split-KV S8 plus merge | 5.767–5.841 / 2.0 | [Scoped pass](raw/operator-split-decode.json), r23 integrated |
| Synthetic joint M12 balanced /skew | 1.0784 / 2.0794, both require 1.3 | [All-cases rejection](raw/operator-moe-joint-m12.json) |
| Actual-route replay rounds 1 /8 | 1.4561 / 1.4435, both require 1.3 | [Routed-stage pass only](raw/operator-moe-route-replay.json) |
| Router existing /reviewed DS /reviewed GLM | 0.5833 / 1.3492 /0.2197, require 2.0 | [All rejected](raw/operator-ordered-router-workers.json) |

Earlier FP4-activation, B-tile, K128/M128, QK8, transient CUTLASS, shared-cache,
HC post and chunk-size failures remain in the
[previous prefill bundle](../2026-09-11-atlas-prefill/RESULT.md). Rejected
configurations stay disabled; narrowing or relaxing a gate after seeing results
does not qualify them.

## Worker process and separate LLooM work

DeepSeek and GLM produced bounded source/helper drafts through LLooM. Original
errors included reversed copy directions, wrong scatter offsets, incomplete
shared tiles, double normalization, invalid aliases and unsupported conclusions
about communication overlap. Independent source review, CPU oracles, real GPU
gates and unchanged thresholds decided what was used. Generic count/profile
helpers were tested on synthetic fixtures; actual routes and traces remained
local. Failed/incomplete worker responses are not passing research evidence.

A separate LLooM Responses bridge release enabled the supervised tool-result,
streaming and usage path; exact commit/artifact hashes and selected verified
facts are in [worker-bridge-release-summary.json](raw/worker-bridge-release-summary.json).
That was a LLooM release, not an Atlas release, Runtime release or model-default
change. The process separated package staging from idle activation so a busy
timeout could not restart active work. Provider prompts, responses, reasoning,
credentials, entity routes and host configuration remain private.

## Reproducibility and unresolved scope

The optional native sparse bridge uses a retained NVIDIA object whose exact
source build has not been reproduced. Source wrappers and an object hash are
not a complete public rebuild of the measured library. Keep this explicit,
retain the source-based BF16 fallback, and see
[next steps](../../../research/atlas/NEXT_STEPS.md).

The latest source includes r24 diagnostics, normally disabled; latest performance
evidence is r23. [Engine lineage](engine-lineage.json) records every retained
source revision; build and source receipts bind the measured artifacts. Model outputs vary across unchanged repeated canaries, so neither
synthetic bit identity nor coherent text establishes broad quality parity.
No alternating appliance matrix, full tools/reasoning/multimodal/cancellation
suite, near 32K-per-owner stress, endurance or release qualification is claimed.
The original raw archive remains private, including model-derived route arrays,
full traces, compiled artifacts and provider envelopes. No weights or binaries
are in this bundle.
