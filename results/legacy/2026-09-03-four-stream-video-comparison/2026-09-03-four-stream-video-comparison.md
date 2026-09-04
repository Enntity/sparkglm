# Four-stream 16K video comparison — 2026-09-03

> **Superseded deployment result:** later live inspection found that this
> SparkGLM arm retained a profiling-only 10 GiB manual KV-cache override. It
> admitted only two of the four streams. The corrected appliance, root cause,
> exact replay, and replacement video are recorded in
> `2026-09-03-c4-capacity-fix.md`. This document remains the receipt for the
> original observation; it is not a verdict on the corrected engine.

## Verdict

The fresh visual A/B does not show a general SparkGLM win. SparkGLM gets the
first two users to visible output materially sooner and is well ahead midway
through the replay, but it then leaves the last two prefills waiting too long.
Mia admits all four streams sooner, finishes 19.28 seconds earlier, and wins
both full-window delivery and post-first-token aggregate throughput.

This is one visual run, not a variance-qualified benchmark. It is sufficient
to expose the current head-of-line shape and to reject any claim that the
current engine obviously dominates Mia on four staggered 16K prose requests.

## Exact comparison

- Hardware: the same two DGX Spark GB10 systems, TP2, used sequentially
- Model: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- Draft: DFlash2 k=7, draft TP2
- Target KV: FP8
- Scheduler: 7,168 maximum batched tokens, four sequence slots,
  mixed-prefill policy `0`
- Workload: four unique prose prompts, arriving at 0/1/2/3 seconds
- Actual prompt tokens: 15,804 / 15,807 / 15,807 / 15,806 in both arms
- Generation: 400 required tokens per request, temperature zero, thinking off,
  EOS ignored
- Prompt salt: `video-ab-20260903` in both arms
- Capture: real streamed SSE timestamps and final API usage
- SparkGLM: image
  `sha256:20d3833c8e3e4b57d5dc74bc3f1c0fced820dfcbca605625790b3ff7e32dd4c4`,
  engine commit `495f27d946`
- Mia: current upstream/image commit `eb0469f`, fetched immediately before the
  run; image `lloom-glm53-mia-eb0469:exl3`
- Both runtimes completed their own 20/20 boot-shape warmup before capture.

## Results

| Metric | SparkGLM | Mia |
| --- | ---: | ---: |
| First-token clock, stream 1 | **27.83 s** | 36.38 s |
| First-token clock, stream 2 | **36.17 s** | 51.65 s |
| First-token clock, stream 3 | 75.95 s | **64.05 s** |
| First-token clock, stream 4 | 100.88 s | **70.38 s** |
| Full wall time | 131.54 s | **112.26 s** |
| Full-window delivery | 12.16 tok/s | **14.25 tok/s** |
| Post-first-token aggregate | 15.39 tok/s | **21.03 tok/s** |

All eight requests completed 400/400 tokens without an API error. At the
80-second replay point SparkGLM has completed stream two and emitted more total
text, while Mia has all four streams producing output. Mia's fairer admission
then overtakes SparkGLM and completes the set first.

The aggregate metric starts at the first visible token, so SparkGLM's earlier
first response creates a longer measurement interval and magnifies its late
stream stalls. Full-window delivery tells the same underlying story more
conservatively: Mia is 17.2% faster by completed tokens per wall second.

## Artifacts

- `glm53-four-stream-16k-sparkglm-495f27d946.json`
- `glm53-four-stream-16k-mia-eb0469f.json`
- Side-by-side video generated at 3,200 x 900, 12 fps, with SparkGLM on the
  left and Mia on the right.

The renderer now labels the footer with actual server-reported prompt tokens
per stream. The displayed speed is reconstructed from captured SSE timestamps;
the video does not synthesize token timing.

## Consequence

The next scheduler gate should target this exact staggered shape. Preserve
SparkGLM's lead for the first two requests while preventing streams three and
four from waiting behind the early decode/prefill sequence. Any candidate must
beat Mia's 112.26-second wall and 70.38-second last first-token clock on a
multi-run, alternating-order A/B before we claim a real appliance lead.
