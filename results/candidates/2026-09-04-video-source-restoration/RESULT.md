# Posted-video source restoration

## Objective and correction

Restore the actual implementation identified by the posted video's
`4b23759+c805318`, not a similarly configured successor. The earlier `b3d0bbd`
reconstruction omitted inherited native FP16 selector/DeepGEMM code and
substantive Python indexer, scheduler, cache-retention, and FA discovery code.
Close timing did not excuse that omission. Its old measurements remain intact
and now explicitly state that limitation.

The source restoration is `207709ab87c1dbe3d23c3bad764758784a4d3a6b`;
`672df9e4155fa1fa12f6dc63dbd41f0fa7272ab7` fixes fresh-source generation with
a reused native build cache. The latter changes build orchestration, not
runtime algorithms. `docs/VIDEO_SOURCE_RESTORATION.md` and the provenance
manifest state the exact boundary.

## Source and operator checks

- All 2,423 installed vLLM Python files match the retained video image, with
  three byte-pinned nonfunctional exceptions: build metadata, EXL3 licensing,
  and RMSNorm comments.
- Eight changed native-source inputs match the historical foundation.
- Six EXL3 compile inputs match the historical recipe, allowing only corrected
  license headers. The actual historical inputs were under `/opt/glm53/pair/`.
- The running-profile source check requires the video's exact 16 ms spinwait
  transformation, rather than allowing arbitrary startup modifications.
- Thirty native FP16/FP32 decode/prefill cases pass exact selected-value,
  unique/in-range-index, and three-replay CUDA-graph checks. Selected-value
  hashes match the historical image in every case.
- The later public chat template differs as a file, but renders exactly the
  same prompt bytes as the historical template for all four video requests.
  Their historical request-body and rendered-prompt hashes are retained.
  The replay client now adds an optional `cache_salt` field; this isolates KV
  reuse between runs without changing the prompt generator or rendered tokens.
- The installed FlashInfer and ExLlama source audit covers another 4,997
  files: only four corrected SPDX headers differ. The installed Torch,
  Triton, FlashInfer, and ExLlama package versions also match.

These are focused operator checks, not the full production-dimension,
boundary, and sanitizer G1 suite. The bundle conservatively retains G0 as its
qualification level; no release promotion is implied by source identity.

## tinyGLM comparison

Both images run the same two-layer, 16-expert dummy fixture on TP2, without
speculation or vision. Each arm discards two complete workload repetitions
before retaining three. Every output token-ID hash matches between the arms.

| tinyGLM case | Reference median tok/s (3 runs) | Initial candidate (3 runs) | Follow-up candidate (10 runs) |
| --- | ---: | ---: | ---: |
| C1 decode | 297.602 | 282.632 | 288.181 |
| staggered C4 | 425.994 | 420.746 | 424.379 |
| long staggered C2 | 87.470 | 86.614 | 86.792 |

The initial C1 result missed the 5% throughput guard by approximately 0.03
percentage points. That failed result is retained. The longer follow-up passes
the same guard in every case; it does not erase the initial result or establish
that the restored implementation is faster. No full-model throughput can be
inferred from these numbers. Other CPU activity remained on the appliance, and
the arms were not interleaved into a publishable paired experiment.

## Full-model replay

The clean image is
`sha256:730b15d4a094131d29c032a74660236151ae67dd33f49dc7bb1e6b6098d1ce66`,
with source revision `672df9e4155fa1fa12f6dc63dbd41f0fa7272ab7`. Both ranks
were checked after startup. This is the real EXL3 model, not dummy weights.
It passed the 20-request shape warmup and the long C4 capacity warmup, then
discarded one complete video-workload replay before retaining three.

| Restored build run | Wall seconds | Aggregate delivered tok/s |
| --- | ---: | ---: |
| Warmup (discarded) | 100.250 | 21.455 |
| 1 | 90.678 | 22.770 |
| 2 | 92.762 | 22.720 |
| 3 | 89.882 | 23.760 |
| Retained median | 90.678 | 22.770 |

Every request completed without error with 400 output tokens. Actual prompt
counts exactly match the video: 15,807 / 15,810 / 15,810 / 15,809. Arrivals
remain 0/1/2/3 seconds, temperature zero, with the original field-guide prompt
salt. The client is `e0aaf9d` (full revision recorded in qualification metadata).
Each repetition uses a separate cache namespace because the production
profile does not expose `/reset_prefix_cache`; prompt text is unchanged.

The retained video's single sample was 86.149 seconds and 24.267 aggregate
tok/s. The restored median is 6.17% lower on that throughput convention. This
is not a speed-parity result. Unrelated CPU activity remains on the appliance.

The retained original image was then run with the same current safe launcher,
checkpoint files, settings, client, prompts, cache-isolation policy, and
warmup sequence. Both ranks' original image IDs were checked; the original
running Python inventory passes without any allowed file differences.

| Retained original image run | Wall seconds | Aggregate delivered tok/s |
| --- | ---: | ---: |
| Warmup (discarded) | 90.652 | 23.091 |
| 1 | 90.622 | 23.375 |
| 2 | 86.690 | 24.023 |
| 3 | 87.552 | 24.126 |
| Retained median | 87.552 | 24.023 |

All twelve measured reference requests also complete with the same exact
prompt counts and 400 output tokens each. The restored median takes **3.57%
longer to finish**, with **5.22% lower** video-style aggregate throughput than
the contemporary retained image. The aggregate metric's denominator starts
at the first visible token, so it need not move in proportion to total wall
time. Report both metrics, not whichever looks more favorable.

This is a slower point estimate, **not certified performance parity**. It
must not be dismissed as noise simply because the ranges overlap. The arms
were collected consecutively (restored three, then original three), not
alternated; the source-equivalent rebuild's remaining performance difference
has not been causally isolated. No runtime optimization was added to disguise
that unresolved result. A controlled alternating campaign and matched native
operator timings are the next discriminating checks, not more recipe knobs.

A CPU-only native-library inspection finds the same reported embedded CUDA
target list in both images, but different host-compiler metadata and binary
hashes. The reference reports Red Hat GCC 13.3.1/8.5.0; the rebuild reports
Ubuntu GCC 13.3.0. The reference native library also differs from the pinned
base library. See `raw/native-build-identity.json`. This establishes a remaining
build-provenance boundary, not a demonstrated cause of the speed gap.

The visualizer's historical aggregate metric uses 1,596 post-first-token
tokens divided by the interval from the first visible token to the final
completion. It includes mixed-work stalls and is not a pure kernel decode
rate. These partial C4 results do not satisfy the full paired G3 matrix.

### Reproduce this specific historical workload

After starting the recorded profile and completing its two startup warmups,
run this on the head host for each image. Change `ARM` to keep the cache
namespace distinct across arms and campaigns; do not change the prompt salt.

```bash
ARM=restored-unique-campaign
ENGINE_COMMIT=672df9e4155fa1fa12f6dc63dbd41f0fa7272ab7
for RUN in warmup r1 r2 r3; do
  python3 benchmarks/four_stream_video.py capture \
    --base-url http://127.0.0.1:8890 --model glm-5.3-flash-exl3 \
    --streams 4 --prompt-tokens 16000 --prompt-style field-guide \
    --prompt-salt current-best-video-20260903-r1 \
    --cache-salt "$ARM-$RUN" --stagger-ms 1000 --max-tokens 400 \
    --recipe-label "$ARM mixed=0" --engine-commit "$ENGINE_COMMIT" \
    --output "logs/$ARM-$RUN.json"
done
```

Inspect each receipt for errors, 400 output tokens per stream, and the actual
prompt counts above. The collector retaining a JSON file is not by itself a
passing test. The port shown is the comparison endpoint; use the operator's
actual endpoint if different. This intentionally preserves the original
15.8K-token video requests rather than silently replacing them with the
methodology's separately calibrated 16,384-token G3 cases.

## Correctness

The native oracle deliberately does not require identical index ordering or
tie-breaking among equal FP16 scores. Those are not guaranteed by native
top-k. It does require the exact multiset of selected values, valid unique
indices, and stable selected values across graph replays. The tinyGLM gate
separately requires exact output token IDs.

The full-model streamed text is **not** identical to the posted sample, nor
identical across the three restored-image repetitions despite temperature
zero. All requests complete, but that is not a semantic equivalence proof.
The source audit and dummy-model token-ID gate must not be cited as proof of
bitwise deterministic full-model generation. The cause of full-model output
variation has not been isolated by this workload.
The retained original also changes text across its repetitions and does not
reproduce the posted text exactly. Thus non-repeatability is observed in both
arms; these receipts do not establish a new rebuild-only quality regression
or quality equivalence.

## Limitations

This is not binary reproducibility, broad model-quality evaluation, or a
complete G3/G4/G5 qualification. The modern public chat template and launcher
safety work are retained explicitly; the historical container is not
impersonated by assigning it the new build's identity.

Both full-model arms use the historical automatic `GPU_MEM_UTIL=0.87`
policy, not a forced identical byte-sized pool. Startup reports 18.86 GiB KV
for the restored arm and 18.24 GiB for the retained-image arm. Both must pass
the four-resident-stream capacity warmup. Other CPU activity and variable
unified-memory availability prevent treating this campaign as an isolated,
alternating G3 experiment.
