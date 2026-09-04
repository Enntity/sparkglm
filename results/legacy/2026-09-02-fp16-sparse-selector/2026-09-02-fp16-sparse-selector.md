# Native FP16 sparse selector acceptance — 2026-09-02

## Verdict

Accept for the opinionated GLM-5.3 Flash two-Spark appliance.

The candidate keeps the sparse-indexer score matrix in FP16 and dispatches it
directly to native FP16 top-k kernels, including GLM-5.3's four-token pooled
indexer path. This removes the FP32 materialization and selection tax from the
active 16K/32K sparse-prefill path. The paired real-workload run improves
staggered C2 wall time by 4.85% at 16K and 3.18% at 32K. All requests completed,
retained their own isolation marker, and contained no marker from a peer.

This is a useful bandwidth win, not the fundamental sparse-prefill fusion. The
remaining largest lever is to stop materializing the full score matrix at all.

## Artifact and frozen configuration

- Candidate image: `sparkglm-vllm:fp16-indexer-candidate`
- Image digest: `sha256:107d899515629f168fdd5f7bac3e967bc3d0fe768124bafd9322c7a3a8c4adc4`
- Image source revision: `6a49e49e7e6a3226197a2ceefcf217cdf55f751e`
- Hardware: two DGX Spark GB10 systems, TP2 over ConnectX-7
- Model: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- KV cache: FP8
- Scheduler: 2,048-token budget, phase interleave enabled
- Speculation: DFlash2, seven draft tokens, draft TP1
- Workload: 128 forced output tokens; C2 starts separated by five seconds
- Only A/B variable: `--attention-config.indexer_logits_dtype` set to
  `float32` for control and `float16` for candidate

The launcher was also corrected to preserve a caller-supplied `EXTRA_ARGS`
value across `.env` loading. Engine logs were checked after both launches and
explicitly reported the intended `indexer_logits_dtype`.

## Paired same-prompt real-workload A/B

The candidate and control used the same `fp16-r2` salt variant. Prompt SHA-256
values matched between modes. Positive percentages below mean the candidate is
faster.

| Case | FP16 TTFT (s) | FP32 TTFT (s) | TTFT delta | FP16/FP32 aggregate prefill (tok/s) | Prefill delta | FP16/FP32 wall (s) | Wall delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Medium C1 | 15.72 | 16.65 | +5.56% | 1005.2 / 949.3 | +5.88% | 20.98 / 20.29 | -3.43% |
| Medium C2 | 17.78 / 30.11 | 17.72 / 31.32 | -0.29% / +3.88% | 900.1 / 870.0 | +3.46% | 38.48 / 40.44 | +4.85% |
| Large C1 | 31.44 | 31.65 | +0.65% | 1023.2 / 1016.6 | +0.65% | 37.10 / 36.77 | -0.91% |
| Large C2 | 35.28 / 62.85 | 36.95 / 65.14 | +4.52% / +3.52% | 948.3 / 917.3 | +3.37% | 72.10 / 74.48 | +3.18% |

The C1 wall figures include noisy 128-token decode tails and are not the
acceptance metric for this prefill-only change. No speculative prediction was
accepted in either run, and decode throughput varied independently of the
selector path. The consistent C2 queue-clearance and aggregate-prefill gains
are the relevant evidence.

## Repeatability check

An earlier independently salted run showed the same direction at the important
large shapes:

| Case | FP16 TTFT (s) | FP32 TTFT (s) | TTFT delta |
| --- | --- | --- | ---: |
| Medium C1 | 17.95 | 16.88 | -6.31% |
| Medium C2 | 18.25 / 33.03 | 17.79 / 34.63 | -2.57% / +4.61% |
| Large C1 | 31.64 | 36.37 | +13.01% |
| Large C2 | 35.34 / 63.01 | 38.62 / 66.32 | +8.48% / +4.99% |

The medium-C1 reversal shows normal whole-engine run variance and is why the
change is accepted on the paired C2 target and kernel evidence rather than a
single headline number.

## Native kernel microbenchmark

`benchmarks/sparkglm/sparse_selector_microbench.py` validates selected values
exactly against `torch.topk` in each input dtype and then times the active
`torch.ops._C.top_k_per_row_prefill` operation. GLM-5.3 pools four KV positions
per score column, so 4,096 and 8,192 columns represent 16K and 32K contexts.

| Rows | Score columns | FP32 mean (us) | FP16 mean (us) | Kernel delta | FP32/FP16 matrix |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4,096 | 29.171 | 26.880 | +7.9% | 8 / 4 MiB |
| 512 | 8,192 | 41.286 | 34.059 | +17.5% | 16 / 8 MiB |
| 2,048 | 4,096 | 137.482 | 86.646 | +37.0% | 32 / 16 MiB |
| 2,048 | 8,192 | 290.491 | 164.091 | +43.5% | 64 / 32 MiB |

The selector returned in-range `int32` indices and exactly the same selected
values as `torch.topk` over the same FP16 or FP32 input. Decode-sized C1/C2 row
counts were neutral, as expected for a prefill optimization.

## Quality boundary

All twelve live requests across the candidate repeat and paired control
completed 128/128 tokens with `finish_reason=length`, retained their own
request marker, contained no foreign request marker, and reported no error.
Output hashes are not expected to match because the deployed generation config
uses stochastic sampling (`temperature=1.0`, `top_p=0.95`).

FP16 scores can reorder extremely close candidates relative to FP32 before
top-k. That is the intended numerical tradeoff and remains separately
configurable. This gate proves operational and semantic health on the frozen
workload; it is not a general model-quality evaluation.

## Next lever

The score tensor still exists. At 32K and 2,048 query rows it is 32 MiB per
sparse layer and rank even after this change. The next major optimization is an
SM121-specific fused scorer-plus-selector pipeline that retains only top-k
candidates in registers/shared memory and never writes or rereads the complete
score matrix. That work must preserve GLM-5.3's kpool=4 positional semantics and
pass the same isolation matrix before acceptance.
