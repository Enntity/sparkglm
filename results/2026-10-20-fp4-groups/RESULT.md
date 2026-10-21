# SparkGLM TP2: corrected FP4 shared and dense groups

2026-10-20. Focused, uncapped comparison on ennspark01/02. Current corrected-FP4 KDA Spark mode with native MLA fusion is the control. Shared FP4 is tested first; dense FP4 is then tested on top of the accepted configuration. Strict native formats are unchanged.

| Configuration | Prose tok/s | Code tok/s | Structured tok/s | Quality |
|---|---:|---:|---:|---:|
| control | 38.004 | 67.970 | 89.873 | 6/6 |
| shared | 39.696 | 68.969 | 91.636 | 6/6 |
| both | 38.231 | 68.207 | 94.544 | 6/6 |

Medians of two repetitions. Every repetition:

| Configuration | Workload | Run1 | Run2 |
|---|---|---:|---:|
| control | prose | 38.449 | 37.558 |
| control | code | 69.172 | 66.768 |
| control | structured | 88.840 | 90.905 |
| shared | prose | 39.511 | 39.880 |
| shared | code | 68.962 | 68.976 |
| shared | structured | 90.515 | 92.756 |
| both | prose | 37.093 | 39.368 |
| both | code | 68.323 | 68.092 |
| both | structured | 94.883 | 94.206 |

## Decision

Selected managed Spark policy: **shared**.

The predeclared quick acceptance rule was all six quality cases passing, all timed completion gates passing, at least 2% geometric-mean throughput improvement across the three workloads, and no workload regression greater than 2%. This is a focused qualification threshold, not a statistical confidence claim.

- shared versus control: +2.62% geometric-mean speed; prose +4.45%, code +1.47%, structured +1.96%. Quality passed: True. Accepted: True.
- both versus shared: -0.58% geometric-mean speed; prose -3.69%, code -1.10%, structured +3.17%. Quality passed: True. Accepted: False.

## Quality and limits

The same six existing prompts were used for every arm: invoice arithmetic, exact JSON extraction, two Python functions with bounded fixture execution, and the original 70K/113K two-hop retrieval cases. Four short cases ran concurrently; the two long cases ran individually. Greedy decoding, Low reasoning, and 8192 output tokens avoided the historical 24-token truncation problem. Full responses, grades, token usage, request hashes and finish reasons are retained.

This is a small smoke test. It does not establish broad quality parity, bit-identical outputs, or that RNE alone fixes the old all-group fidelity incident. These switches change both prefill and decode. Shared prefill can use an FP8 cache expanded from the requantized weights; dense prefill also uses the requantized weights.

Both conversions use the existing corrected GLM weight quantizer, quantize_bf16_to_nvfp4_weight_rne. No quantization kernel or activation quantizer changed. Spark mode remains selected throughout, preserving the corrected DFlash quantizer. MLA and KDA prefill stay native. Each arm uses a fresh isolated prefix/swap cache root.

Timing uses the same existing RigMark prose/code/structured prompts and the first two saved nonce strings, seed 20260905, temperature 0, top_p 1, Low reasoning and 4096-token allowance. Decode excludes prefill; different generated text and speculative acceptance can influence measured rates. Both ranks use 4x1m, memory utilization 0.91, KV sharding, display carveout and 48 GiB prefix-cache allowance. GPU clock-cap services were disabled throughout timing and restored afterward.

## Validation and live identity

- Three targeted Rust mode tests passed, including all shared/dense combinations and Strict rejection of nonnative policies. Product launch tests, converter tests and shell/benchmark-driver checks are recorded in validation.json.
- Changed Rust source passes formatting. Workspace-wide formatting reports a pre-existing wrap in model/types.rs from native-MTP work; it was left untouched. No full benchmark or full engine test campaign was run.
- Engine: `e1ee623547ee8ca85abb4e17091907cb267175fe`.
- Binary: `28e19acec1b26afcfbb11428b587d56e80a57504cd77e9cd32669e98e00e4c79`.
- Image: `sha256:f5c00e667bd2473ec26a243a3c2d28573a8986bbe8a8063f078b30e5f0051a20`.
- Product commit: `75b63a8a02986022c2a078f6eb06be58becf5c1d`.
- The final image contains the same tested binary; only accepted product mode defaults were packaged afterward. Both managed ranks have been checked, normal clocks restored, LLooM lease released, and the gateway answered 42. Qwen03/04 was not modified. This is a local unpublished RC.

- The first managed restore was refused by LLooM memory admission. A normal retry succeeded; no memory budgets, runtime estimates, or container limits were changed. The failure receipt is retained.
- All 201 effective Atlas settings were compared between the accepted experiment and managed serving. Precision and optimization settings match; the managed launch lists the same two RDMA devices in the opposite order. Normal serving restores the clock caps, so the uncapped numbers describe the measurement condition.
- All six control outputs match the preceding MLA-fusion RC in text, reasoning hashes, token counts and finish reasons. The public Spark preset rejects dense-FP4 overrides because that candidate did not qualify.

Raw receipts and scripts are under raw/. SHA256SUMS covers the bundle.
