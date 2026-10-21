# SparkGLM TP2 RC qualification

**Targeted sampling repair: PASS. Overall RC qualification: FAIL.** The native-MTP sampling bug was repaired with one production-code guard. Completed unaffected validation is retained with its original build identity. The capacity probe is a separate repair-stage gate; its failure does not mean the sampling repair failed.

Release blockers:

- Spark scored 39/40 arithmetic cases versus Strict 40/40. The original Spark-only miss remains unresolved; the later C1 diagnostic does not replace it.
- Near-limit native capacity: Four overlapping client requests were admitted one at a time: KV admission counted 65462 committed blocks of 71237 and queued all remaining requests.
- Near-limit native capacity: Rank 0 minimum available memory 3.776 GiB was below the declared 4 GiB requirement.

## Repair and retained validation

The masked-chat verifier used a raw argmax shortcut for nonzero-temperature native-MTP requests. It now declines that shortcut unless sampling is greedy. A regression fails on the old code; ten verifier-helper tests pass after the repair. Native libraries, profiles, launcher and weights are unchanged.

The repaired native arm completed 960 independent draws against the preserved 960-draw serial control. No distribution difference was detected (worst adjusted p=1.0); this is limited statistical evidence, not proof of equivalence. C1 and C4 greedy results match all eight original prompts exactly. The non-streaming chat assertion was corrected from saved receipts without new inference; its original failure remains in the bundle.

| Tested policy | Strict | Spark |
|---|---|---|
| Routed decode activations | A16 | Calibrated A4 |
| KDA decode weights | Native MXFP8 | NVFP4 with corrected nearest-even rounding |
| Shared-expert weights | Native FP8 | NVFP4 with corrected nearest-even rounding |
| KDA prefill, dense projections, vocabulary head | Native formats | Native formats |
| MLA projections | Native FP8 with fusion | Native FP8 with fusion |
| Speculation | Optional native MTP; serial in Kindling | DFlash2 |
| KV cache | FP8 | FP8 |

| Retained check | Strict + native MTP | Spark + DFlash2 |
|---|---:|---:|
| Behavior supplement | 30/30 | 30/30 |
| Tool cases | 14/14 | 14/14 |
| Arithmetic | 40/40 | 39/40 |
| Short two-hop retrieval | 12/12 | 12/12 |
| Long two-hop retrieval, through about 113K | 60/60 | 60/60 |
| Prompt NLL, 25,050 scored tokens | 0.959750 | 0.962332 |

All quality and retrieval responses stopped naturally. The Spark arithmetic miss returns -382130 instead of -381530. The same error exists in older evidence, but that does not establish a matched fully native TP2 baseline or clear the current conversion policy. The declared paired-quality gate remains in force.

## Decode and serving performance

RigMark medians of two runs, tok/s. Both GPUs were uncapped. Each of the six decode responses per mode stopped naturally within the 4096-token allowance. These measurements belong to the frozen image; the repair does not change the greedy path.

| Mode | Prose | Code | Structured |
|---|---:|---:|---:|
| Strict + native MTP | 31.211 | 41.133 | 50.364 |
| Spark + DFlash2 | 38.581 | 66.543 | 90.136 |

Saved historical best NVFP4 RigMark medians are 39.344 / 72.252 / 91.944 tok/s for prose / code / structured. Current results use fresh nonces and two repetitions; the older best code result also used 4x512k. These are historical references, not a newly matched A/B. No old-build run was repeated.

Spark cold prefill medians: 2749 tok/s at 8K, 2849 at 32K and 3008 at 64K. Warm replay TTFT medians: 179, 182 and 198 ms. C4 aggregate throughput is 75.868 tok/s for the separate 256-token code workload; it is not an uncapped-output result.

Kindling on the repaired image used 8x128k at .91, 80 measured runs and 193 streams per mode including warmups. GPU clocks were uncapped; outputs had the stock 512/768-token limits. Both modes used the same explicit zero-token thinking budget, Low effort and fresh nonce series. Strict retained its loaded server after the failed harness attempts; the fresh nonce series avoids reusing those prompt prefixes. Spark used a fresh load. The earlier legacy-flag partial run and rejected top-level-zero warmup are retained separately. Fixture texts and timing calculations are unchanged; these adapted requests are not interchangeable with historical stock Kindling scores. Mixed C4/C8 numbers are aggregate batch throughput; the other rows are single-stream decode throughput.

| Kindling workload | Strict serial tok/s | Spark tok/s | Failed runs, Strict / Spark |
|---|---:|---:|---:|
| prose | 25.759 | 42.194 | 0 / 0 |
| code | 25.573 | 70.959 | 0 / 0 |
| structured | 25.685 | 91.054 | 0 / 0 |
| counting | 25.692 | 98.376 | 0 / 0 |
| mixed4 | 25.209 | 80.283 | 0 / 0 |
| mixed8 | 25.184 | 95.081 | 0 / 0 |

## Arithmetic diagnostic

Two additional C1 requests compared Spark speculative decoding with its serial target on the repaired 8x128k load. Token streams matched: **True**. The expected answer is -381530. The original failing check used C4 and 4x1m, so this diagnostic does not replace that result.

Speculative:

```text
2123 × 49 = 2123 × 50 − 2123 = 106150 − 2123 = 104027
7256 × 67 = 7256 × 70 − 7256 × 3 = 507920 − 21768 = 486152
104027 − 486152 + 595 = 104622 − 486152 = −381530

Answer: -381530
```

Serial:

```text
2123 × 49 = 2123 × 50 − 2123 = 106150 − 2123 = 104027
7256 × 67 = 7256 × 70 − 7256 × 3 = 507920 − 21768 = 486152
104027 − 486152 + 595 = 104622 − 486152 = −381530

Answer: -381530
```


## Near-limit capacity

Capacity probe passed: **False**. Tokenizer-counted prompt: **1,046,988 tokens**. Native MTP used .92 utilization with the existing 4096 MiB guard. The old .91 allocation could hold only 1,014,752 target tokens despite the configured 1,048,576 limit.

Both cold answers were exact. Native-produced prefixes restored correctly, and all four client requests returned the same tokens as the seeded extension. However, KV admission ran those four requests one at a time. Rank 0 available memory reached 3.776 GiB, below the declared 4 GiB requirement. Near-limit cancellation/reuse did not run after the failure. This does not qualify native MTP for simultaneous C4 at the maximum context.

Cache accounting caveat: the first native request after a serial-produced prefix reported cached tokens while server logs showed full recomputation. Later native-produced prefix hits were verified from restore logs and subsecond TTFT.

## Build and restoration

- Engine: `a628b2aed4efe515489a7b7528a5ec9fb79f66a9`.
- Image: `sha256:9042aba369f2ee1b45bdec4df733a82826524cd0184abac4ae97836cf894cdf9`.
- Binary SHA256: `41780770870a9509262f31a6f891e81ec8cfdf9dd3ccfd08a83fc5d17e8db898`.
- Both test containers were removed, clock caps restored, the LLooM lease released, and managed Spark serving verified on both ranks. The gateway answered 42.
- Managed serving remains on the pre-repair Spark image `sha256:f5c00e667bd2473ec26a243a3c2d28573a8986bbe8a8063f078b30e5f0051a20`. No new public release was published.

The JSON assessment records every retained and superseded failure. Raw receipts, identities, server logs and the surgical source patch are in the verified archive. Kindling source code is not included. Strict preserves checkpoint weight formats and native activation policy; both modes still use the engine kernels and FP8 KV cache, so no reference-engine bitwise-equivalence claim is made.
