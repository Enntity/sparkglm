# NVIDIA NVFP4: 262K context / 12 GiB KV per rank

The requested configuration loaded, passed the bounded C4 and near-limit context
screens, and recorded zero preemptions across warmup, all three captures,
semantics and the context probe. It remains experimental, not a new default.

| Configuration | Run 1 s | Run 2 s | Run 3 s | Median s |
| --- | ---: | ---: | ---: | ---: |
| baseline | 90.500 | 93.658 | 82.103 | 90.500 |
| candidate | 85.271 | 79.578 | 93.707 | 85.271 |

Baseline is 524288 context / 9 GiB KV per rank; candidate is 262144 / 12 GiB.
Observed median wall change: **-5.78%**. This is a 5.8%
reduction in total wall time, not a statistically established speedup. The
baseline was captured the previous night rather than rerun or alternated.
Both knobs changed, so this does not isolate their individual effects.

Same NVIDIA target revision, DFlash2 MXFP8 TP2 depth 7, FP8 KV, native CUTLASS,
2048-token prefill chunks, four sequences, serving images and prompt hashes.
Traditional four field-guide streams arrived at 0/1/2/3 seconds with 400 output
tokens each; actual prompt counts were 15807, 15810, 15810 and 15809. One exact
warmup was discarded per arm. Every retained request completed without error.
See [all values](comparison.json), [environment](environment.json) and [raw](raw/).

Median TTFT by request was 10.85/19.63/28.50/50.93 seconds at 9 GiB and
13.81/22.77/31.74/44.10 seconds at 12 GiB. The fourth request improved while
the first three worsened. Earlier baseline counters recorded 39 preemptions
across warmup, three captures and semantics. Candidate per-capture counters
stayed at zero, and its final endpoint counter remained zero after the context
probe. Counter scopes are identified; no baseline per-run count is invented.

## Near-limit context and semantics

A 261120-token cold prompt returned the correct START-code tool call with
198.62s TTFT; two cached continuations returned the correct MIDDLE and END codes
at 3.86s and 3.37s TTFT. All three finished with `tool_calls`, correct arguments,
and no request error. This is one near-limit single-client probe, not proof of
four concurrent full-window requests or endurance.

The bounded semantic suite remains 14/16, failing the same two arithmetic
questions (123*47 and 91*89) as the baseline NVIDIA package. No model quality
improvement is asserted.

## Memory and limits

Two-second host samples, from startup through the context probe, recorded
minimum available RAM of **2.05 GiB head** and
**2.53 GiB worker**. A per-node guard would stop only the
experimental container after three consecutive samples below 1.5 GiB; it never
triggered. Sampling can miss faster allocation peaks and is not an OOM guarantee.

Host swap usage rose during this session, peaking at
**8.42 GiB head** and **6.55 GiB worker**.
This is total host swap, not proof that model tensors were swapped, and swap
occupancy is not a paging-rate measurement. The memory tradeoff is material;
zero preemptions do not imply unlimited host-memory headroom.

Startup reported 597385 effective cache tokens and 2.28x maximum concurrency at
262144 tokens, versus 662356/1.26x for the old configuration. These hybrid-cache
figures depend on the context/configuration and are not a flat pool to compare
by token count alone. The observed C4 preemptions are the useful result here.

No cold-start repetition, long C4 capacity, broad quality, partial-node failure
or endurance qualification was performed. The 9 GiB source default remains
unchanged. Q38FN restoration is tracked separately in the private operations
receipt; inference tests were isolated from Presence after verifying cloud routing.
