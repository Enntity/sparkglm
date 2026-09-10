# NVIDIA NVFP4: 512K context / 12 GiB KV bounded stress

**Not reliable enough as an appliance at 512K/12 GiB.** The initial bounded
stress run completed capacity and serving reliability screens with zero endpoint
preemptions, OOMs or engine failures. A subsequent managed reload of the same
image pair, target, context and cache setting dropped head available RAM to
**0.80 GiB** and triggered the 1.5 GiB guard before API readiness. This supersedes
the initial favorable assessment: the configuration has insufficient cold-start
margin on this host. It is rejected for unattended appliance use at this setting;
the successful performance measurements below remain valid for their run.

The failed reload used the normal gateway-facing managed containers rather than
the temporary direct-backend serving name/port. An earlier attempt with stale
9 GiB worker bootstrap was caught and stopped before full loading; both ranks
were verified at 12 GiB for the recorded failure. No CUDA OOM is asserted: the
memory guard intervened. See [reload receipt](raw/managed-reload.json).

This is not G5 qualification. Defaults remain unchanged.

## Traditional comparison

| NVIDIA configuration | Retained C4 wall times (s) | Median (s) |
| --- | --- | ---: |
| 512K / 9 GiB, earlier baseline | 90.500, 93.658, 82.103 | 90.500 |
| 262K / 12 GiB, earlier screen | 85.271, 79.578, 93.707 | 85.271 |
| **512K / 12 GiB, this run** | **93.135, 84.866, 85.296** | **85.296** |

The observed median is 5.75% shorter than the previous 512K/9 GiB baseline and
essentially identical to 262K/12 GiB. Historical controls were not reloaded or
alternated, so this is not an isolated or statistically established speedup.
All twelve retained requests returned their full 400 tokens without error.
One exact warmup was discarded. Prompts and cache salts match the retained
NVIDIA baseline; arrivals are 0/1/2/3 seconds, with approximately 16K input tokens
per request. Raw captures include actual token counts and prompt hashes.

Same pinned NVIDIA target, DFlash2 MXFP8 TP2 depth 7, native CUTLASS W4A4,
FP8 KV, four sequences and 2048-token prefill chunks. See [environment](environment.json).
Startup reported 884386 effective cache tokens, 1.69x concurrency at 524288,
and 1.74 GiB graph memory. This does not promise four concurrent full windows.

## Capacity, contention and recovery

- **523264-token cold prompt:** correct START-code tool call, 396.26s TTFT.
  Cached MIDDLE and END tool calls were correct at 2.02s and 2.06s TTFT.
  The cold path includes first-use kernel compilation.
- **Four independent 64512-token conversations:** all twelve START/MIDDLE/END
  tool calls were correct. Cold TTFTs were 52.65, 190.42, 98.01 and 146.51s;
  cached follow-ups ranged from 0.84 to 28.99s. These followed the nearly full
  window, exercising cache turnover. Zero preemptions did not eliminate
  scheduling latency.
- **Six simultaneous 16384-token clients:** all six completed 128 output tokens
  without transport errors; four running and two waiting were observed.
  Total wall time was 81.25s. No foreign request marker appeared. One of six
  omitted its own requested marker within the output budget: the strict content
  assertion failed. This is retained as a content failure, not counted as a
  fully passing isolation test, and its cause is not established.
- **Semantics:** 14/16, with the same arithmetic failures (123*47 and 91*89)
  as earlier NVIDIA screens. No quality improvement is established.
- **Cancellation:** stream disconnected after first content, active/waiting
  counters drained within 0.52s. Three subsequent exact replies passed in
  0.85–0.95s. The original probe waited for prose without recognizing the
  `reasoning` event field; it was terminated and a corrected probe rerun.
  Both versions are retained. This was a harness correction, not an engine fix.

The controller stopped at the queue-content assertion; its FAILED event is
preserved. Semantics and corrected cancellation were then run separately.
The final endpoint counter was zero preemptions and zero running/waiting
requests. No traffic was silently discarded to turn the controller green.

## Memory pressure

Two-second per-host sampling included available RAM, swap page counters and
memory PSI. Both hosts use 4096-byte pages. The guard would stop only the test
container after three consecutive samples below 1.5 GiB; neither guard fired.

| Measurement | Head | Worker |
| --- | ---: | ---: |
| Minimum available RAM, startup through shutdown | 1.59 GiB | 3.44 GiB |
| Peak occupied swap | 8.49 GiB | 5.28 GiB |
| Swap read during serving through final recovery | 0.91 GiB | 0.077 GiB |
| Swap written during serving through final recovery | 3.67 GiB | 0.005 GiB |
| Full memory PSI stall time in that serving interval | 12.18s | 1.17s |

These are host-wide measurements, not attribution to model tensors. Serving
paging counters exclude model shutdown. Initialization also caused pressure;
peak full PSI avg10 was 14.71% on the head and 61.23% on the worker across the
whole record. The system recovered and did not continuously thrash, but real
paging occurred. The 1.59 GiB low is uncomfortably close to the guard and
sub-second allocation peaks are not covered by two-second samples.

## Scope and reproduction

This uses the existing capture/context/semantic helpers from the sibling
262K/12 GiB bundle and the AGPL staggered benchmark at
`benchmarks/staggered_openai.py`, with an output-file option added in the private
runner. No serving code, models, scheduler policy or recommended profile changed.
Original cancellation and summary helpers are included here; complete private
launch/controller logs and synthetic request payloads are retained separately.
Selected raw measurements and every public artifact are checksum-bound by
[qualification.json](qualification.json).

Tests used the direct head backend, not the authenticated gateway. No repeated
restart, partial-node failure, long soak, four simultaneous 512K conversations,
or broad task-quality suite was attempted. Local Q38FN and Presence restoration
is operational work verified separately; these results do not claim a product
release or a new default.
