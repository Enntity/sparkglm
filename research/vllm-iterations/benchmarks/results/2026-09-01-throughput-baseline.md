# Resident throughput baseline — 2026-09-01

This is the unchanged resident `sparkglm-vllm:e3f690c` strict-skip engine on
the two DGX Sparks. The run used the fixed medium/large C1/C2 matrix, five
second C2 stagger, 64 forced output tokens per request, and fresh prompt salt
`2026-09-01-throughput-a`. The model stayed resident; no reload or server
restart occurred.

| Case | Prompt tokens | TTFT A / B (s) | Effective prefill A / B (tok/s) | Active decode A / B (tok/s) | Active decode sum (tok/s) | Mixed-window delivery (tok/s) | Wall (s) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Medium C1 | 15,808 | 17.19 | 919.6 | 27.3 | 27.3 | 27.28 | 19.50 |
| Medium C2 | 15,808 / 15,805 | 19.24 / 31.11 | 821.8 / 508.0 | 31.9 / 37.6 | 69.5 | 6.79 | 37.79 |
| Large C1 | 32,180 | 32.28 | 996.9 | 37.1 | 37.1 | 37.11 | 33.98 |
| Large C2 | 32,180 / 32,177 | 35.82 / 60.86 | 898.5 / 528.7 | 27.6 / 32.4 | 60.1 | 3.94 | 67.81 |

All six requests completed all 64 requested output tokens. Maximum visible
token gaps were 0.135–0.154 seconds.

The two decode rates answer different questions. `Active decode sum` adds each
request's post-first-token rate and describes admitted decode work.
`Mixed-window delivery` divides every post-first-token output token by the
interval from the first request's first token until the final request ends. It
therefore exposes strict-skip's long no-output interval while the second prompt
prefills. Both must be sampled after every engine change.

The exact machine-readable matrix is in
`2026-09-01-throughput-baseline.json`.
