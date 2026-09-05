# Ten-repetition preserved-image comparison

Declared before retained fixed-cache results were inspected. This is a focused
reproduction diagnosis, not the complete G3 workload matrix or a promotion.

- Primary metric: total four-request workload completion wall time, lower is
  better. Secondary: aggregate delivered post-first-token throughput, request
  TTFT, visible streaming gaps, completion/error counts and speculative work.
- Ten measured repetitions per image, split into independently restarted and
  warmed blocks of five. Order: rebuilt A1, preserved B1, preserved B2, rebuilt
  A2. Reversing the two comparisons reduces a simple linear order effect; two
  startups per image expose some startup variation. This does not create ten
  independent startup pairs or remove arbitrary time-dependent confounding.
- A1 is the already-started `clean-fixed650` block. Later arms are
  `original-fixed650-b1`, `original-fixed650-b2`, `clean-fixed650-b2`.
- Both ranks must use the declared immutable image digest, CPU set
  `5-8,15-19`, and **650 actual GPU cache blocks / 1,132,404 reported cache
  tokens**. Both images receive `--num-gpu-blocks-override 650`. This is a
  diagnostic control, not a new recommended default.
- Interpretation clarification added after the first block pair, without
  changing the test or analysis plan: matching the pool and CPU placement isolates
  the images under these controls. It does not prove identical performance
  with unconstrained CPU placement and automatic memory sizing, nor does it
  identify which prior uncontrolled factor caused a particular slow run.
- Preserve the exact posted-video workload: four field-guide prompts, target
  16K (actual counts 15,807 / 15,810 / 15,810 / 15,809), arrivals at 0/1/2/3
  seconds, 400 delivered tokens per request, temperature zero, thinking off,
  EXL3 TP2 and DFlash2 k7. Prompt bytes stay fixed; unique cache namespaces
  prevent prefix reuse. This intentionally reproduces the video rather than
  substituting the newer exactly-calibrated G3 fixture.
- Complete shape and long-C4 warmup on every startup, then discard one exact
  replay. Retain the discarded trace for transparency, not inclusion in the
  measured ten. All outliers remain; failures stop the campaign and are
  reported rather than silently replaced.
- Report every run, block and image means/medians/ranges. Inspect the direction
  of both reversed-order block comparisons before attributing a small pooled
  difference to the image. Any per-run confidence interval must be explicitly
  conditional on these four startups, not presented as startup-independent
  evidence of equivalence. Lack of a significant difference is not proof of
  equality. No post-hoc outlier removal or choice of fastest sample.
- GPU clocks/temperature/power and engine work counters accompany each replay.
  Unrelated CPU work and a small idle sidecar are preserved; the appliance is
  not a perfectly isolated laboratory. No known engine optimization is
  changed. Prior automatic-cache runs remain separate evidence.

The collector and continuation driver are original diagnostic orchestration.
Existing engine provenance and upstream licensing remain unchanged. Machine
addresses, access credentials, model files and compiled artifacts are not
part of this source-only result bundle.
