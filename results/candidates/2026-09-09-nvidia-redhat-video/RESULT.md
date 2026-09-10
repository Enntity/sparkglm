# NVIDIA versus Red Hat: traditional SparkGLM comparison video

Fresh captures on the same TP2 DGX Spark pair, using the same serving images,
DFlash2 MXFP8 TP2 depth 7, FP8 KV, 9 GiB KV per rank, configured 524288 context,
four sequences, mixed scheduling and 2048-token prefill chunks. The intentional
change is the pinned target checkpoint and its loader: Red Hat compressed-tensors
versus NVIDIA ModelOpt. Source package: `f4e54e8`; image identities are in
[environment.json](environment.json). No MTP patches are enabled.

| Checkpoint | Run 1 seconds | Run 2 seconds | Run 3 seconds | Median seconds | Output tokens / total wall second |
| --- | ---: | ---: | ---: | ---: | ---: |
| redhat | 77.533 | 89.243 | 89.762 | 89.243 | 17.93 |
| nvidia | 90.500 | 93.658 | 82.103 | 90.500 | 17.68 |

NVIDIA median wall-time change: **+1.41%**.
Negative means faster; positive means slower. These are observed medians, not
a statistically established population speed difference. All 24 retained
requests completed 400 tokens without API errors. Identical prompt SHA-256s
and server-reported input token counts were verified for every corresponding
request, including the discarded warmups. Actual prompt tokens by stream:
`[15807, 15810, 15810, 15809]`.

## Capture and video method

Traditional four field-guide streams, arrivals at 0/1/2/3 seconds, temperature
zero, reasoning disabled, and 400 output tokens each. A unique KV cache salt
per repetition prevents reuse between runs without changing the prompt text.
Each arm has one discarded exact-workload warmup and three retained runs.
Run order is Red Hat's complete block followed by NVIDIA's complete block.
This avoids extra weight reloads, but does not satisfy alternating-pair G3.
The earlier five-case isolation screen is a different workload and remains
[separately reported](../2026-09-09-nvidia-modelopt/RESULT.md).

The top NVIDIA and bottom Red Hat panels each show their median wall-time
capture, at 1x elapsed time with all startup-to-first-token waiting retained.
Actual SSE event times determine text appearance. Animated token counters
apportion server totals across text deltas; they are not measured per-token
arrival times. [All sample values](comparison.csv), [selection receipt](comparison.json),
and [video hash](video.json) accompany the [raw captures](raw/).

## Semantics and limits

Bounded semantic score: NVIDIA **14/16**;
Red Hat **15/16**.
NVIDIA failures: `['arithmetic_123_47', 'arithmetic_91_89']`.
Red Hat failures: `['arithmetic_91_89']`.
These arithmetic checks do not rank general model quality. No quality advantage
is asserted. Operational results and failures are retained rather than converted
into a universal token-identity gate. Target quantization and floating-point
execution can change generated tokens.

Selected counters cover warmup, three captures and the semantic suite together;
do not treat their speculation acceptance ratio as capture-only acceptance.
Cache preemptions and their latency effects are included. Full-window NVIDIA
capacity, endurance, restart repetition and partial-node failure remain
unqualified. This is the maintainer-selected NVIDIA package, not a G5 release.
NVIDIA DFlash2 remains resident on the isolated test endpoint; application
Presence stays on its cloud route and Qwen remains stopped.

The draft's separate CC BY-NC-ND terms still apply. No weights, private host
configuration or generated MP4 is included in the source result bundle.
