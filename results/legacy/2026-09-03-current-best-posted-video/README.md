# Current-best posted four-stream capture

This is the raw timing receipt behind the final current-best comparison posted
on 2026-09-03. It corrects an identity mistake in the initial public-source
staging: the posted capture was not the earlier `e7e35579` corrected-C4 run.

## Identity

- Optimization commit:
  `4b2375977df873be57bc277b17778f94323ef7e6`
- Required launcher follow-up:
  `c80531867e13085c356ae5b9bff4c3b98ee64e8b`
- Image: `sparkglm-vllm:exl3-grouped-prefill-candidate-20260903d`
- Image ID:
  `sha256:0b17bd9246763d74e2f5e1b79fecdcb6a8ef03e1b8e5823f2d2183ceafb91159`
- Target revision:
  `25a44fdbf16862a46b7cc9921142c6c81350af2f`
- DFlash2 revision:
  `7d74cdd881ed7e32c31175984a67823127b66cfe`

The live container was inspected before capture and had mixed scheduling,
grouped prefill, and cooperative decode enabled, with cooperative decode capped
at 16 tokens on both TP ranks.

## Workload and result

- Four independently salted, approximately 16K-token field-guide prompts
- One-second stagger
- 400 requested and delivered output tokens per stream
- 86.148650 seconds wall time
- 24.267091 aggregate delivered output tokens/s
- TTFT by stream: 20.379, 29.439, 38.596, and 42.418 seconds
- No request errors in the retained receipt

The locally preserved clicks-oriented top/bottom render is
`glm53-flash-tp2-four-stream-comparison.mp4`, SHA-256
`3dca189abab2e9350812116b5c7acf73cbe832ae9bc18dfcd44a7f507ce3583c`.
It is a time trim of captured SSE events, not a speed-up of those events. MP4
files are excluded from the source repository.

## Evidence boundary

This was a single warmed visual run produced before the current qualification
policy. It is useful provenance for the advertised build target, but it is not
retroactively certified as a repeated G3 performance comparison or a G4
semantic-quality result. Fresh release-candidate results belong in a new
qualification bundle and must not overwrite this receipt.
