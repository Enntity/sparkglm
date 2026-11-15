<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.5, 2026-11-15: prefill yields to decoding streams

The [2026-11-14 results](../2026-11-14-0.3.5/RESULT.md) left one cause of slow
concurrent streams unfixed: while one request prefills, a stream that is
decoding gets one step per 8K-token chunk, so it freezes for 3 s at a time.
This build fixes that with one more engine change, and it leaves every token
as it was. Measured on one DGX Spark pair (TP-2, Spark mode, profile `4x1m`,
GPU clock capped).

## What changed

A prefill pass now stops between two layers, the scheduler runs its ordinary
decode step, and the same pass continues
([Enntity/atlas#32](https://github.com/Enntity/atlas/pull/32),
`ATLAS_GLM_PREFILL_YIELD_MS`). The pass computes the same layers over the
same rows in the same order, so its output is bit for bit what it was.
`install/serve.py` sets a 300 ms target in Spark mode on a pair. Strict mode
and four Sparks are unchanged.

The engine pin is Enntity/atlas `spark` `59ead299` (tree `dfd866dc`); the
image is `ghcr.io/enntity/atlas-sparkglm:712cf8246f85`, built with
`ATLAS_IMAGE_CHECKS=0`. It has not been pushed.

## What it buys and what it costs

A cold 36K-token prompt arriving beside one decoding stream
([`gbench.py`](../2026-11-14-0.3.5/raw/harness/gbench.py), one run each):

| | 0.3.5 without the yield (11-14) | This build | exlr8 (11-14) |
|---|---|---|---|
| Longest gap in the decoding stream | 2.93 s | **0.52 s** | 0.85 s |
| Time to first token of the new request | **13.2 s** | 16.4 s | 38.7 s |
| End-to-end tok/s of the pair of requests | **33.7** | 33.3 | 22.0 |

Two decoding streams while five prompts arrive
([`raw/harness/yexact.py`](raw/harness/yexact.py)); the same engine source
with the switch off (2026-11-14) against this image:

| Arriving prompt | Longest decoder gap | Its time to first token |
|---|---|---|
| Cold, 40K tokens | 3.02 → 0.54 s | 14.8 → 19.5 s |
| Fork after a shared 4K prefix, 22K | 3.45 → 0.48 s | 8.7 → 11.5 s |
| Warm continuation, 44K | 1.66 → 0.61 s | 2.5 → 3.1 s |
| Cold, 9.8K | 2.85 → 0.48 s | 4.2 → 5.5 s |
| Cold, 5.2K | 2.11 → 0.47 s | 2.2 → 2.9 s |

The cost is the arriving prompt's first token, later by the decode steps that
ran in between. Where cold prompts arrive staggered, the whole wave takes
longer:

| Staggered cold prompts, wall time of the wave | 0.3.5 without the yield | This build |
|---|---|---|
| 1 stream, 16K / 32K | 14.7 s / 20.9 s | 14.6 s / 20.8 s |
| 2 streams, 16K | 24.7 s | 25.4 s (+2.5%) |
| 2 streams, 32K | 35.3 s | 36.8 s (+4.2%) |
| 4 streams, 16K | 45.5 s | 46.8 s (+2.8%) |

The pause itself costs nothing these runs can show. Over the four cold and
fork prompts above, the first token came 9.6 s later in total and the two
decoders emitted 534 more tokens in that time: 55.6 tok/s, the rate two
streams decode at with no prefill running (53.7 to 59.0 in the cells below).
The engine log of the first start says the same: a chunk of 8,192 tokens with
12 decode steps inside it took 4.17 s, which is 12 steps of 93 ms and 3.05 s of
prefill, against about 3.0 to 3.1 s for the chunk alone (2.84 s on a normal
start, scaled by that start's slower prefill). That bounds a pause at under
10 ms ([`raw/harness/pause.py`](raw/harness/pause.py) reads it from
`raw/yield/rel-*.log`).

The wave takes longer because the yield moves tokens out of shared steps.
Tokens a stream decodes while another prompt prefills are decoded alone, and
as many of the other stream's tokens are then decoded alone at the end;
without the yield both sets share steps. One stream decodes at 42 to 44 tok/s
and two at about 28 each, so a token decoded alone costs 23 ms where its share
of a two-stream step costs 18 ms. With about 12 pauses a chunk and a lone step
of about 68 ms (the 2026-11-14 step fit, not measured here), that predicts
0.6 s more for the two-stream 16K wave and 1.6 s more for the 32K one.
Measured: 0.6 s and 1.5 s. The four-stream wave was not worked through. A longer target
moves fewer tokens: 600 ms would about halve this and lengthen the freeze
(not run on this build).

A request with nothing decoding beside it is level. RigMark, one round: prose
39.0, code 67.5, structured 91.3 tok/s, cold 16K prefill 2,857 (without the
yield: 38.7, 67.1, 90.9, 2,851).

Decode with prompts already cached, where nothing prefills, moved both ways
by about as much as the four runs of 2026-11-14 differed from each other
(tok/s combined, 2.4K-token prompts,
[`cells.sh`](../2026-11-14-0.3.5/raw/harness/cells.sh), one run a cell):

| | 1 stream | 2 streams | 4 streams |
|---|---|---|---|
| Free text, without the yield | 46.2 | 56.1 | 69.2 |
| Free text, this build | 43.2 | 53.7 | 69.8 |
| JSON-schema, without the yield | 41.8 | 54.7 | 68.5 |
| JSON-schema, this build | 42.3 | 56.1 | 74.5 |

Two free-text streams read 53.7 and 53.8 on the two starts of this build,
against 54.8 to 56.4 over the four earlier runs. That may be a small cost
and is not resolved.

## Same tokens

- **Yield on against yield off.** The scenario above on a freshly started
  engine, compared with the same engine source built with the switch off and
  given the same requests in the same order. Streamed: all 7 outputs
  identical. Unstreamed with log-probabilities, prompts already cached: 7
  outputs and 341 token log-probabilities identical. Unstreamed, a new set of
  prompts: 7 outputs and 340 log-probabilities identical
  ([`raw/yield`](raw/yield): `foff-*` against `rel-*`;
  `python3 raw/harness/yexact.py --cmp A B` repeats it).
- **Alone against waves.** 12 JSON-schema and 12 free-text prompts, each
  generated alone and in waves of two, three and four: 40 of 40 identical
  each.
- **Against 0.3.4.** Greedy output 8/8 identical; prompt log-probabilities on
  1K, 4.4K and 15K-token prompts zero difference at every token. Live suite
  and guard log 109/109. Thinking-off checks 15/15.
- **Soak**, 180 s of mixed traffic at four streams: no engine errors. 67 of
  68 replies valid; the other was a free-text request that spent its
  250-token limit reasoning, as on 2026-11-14.

A comparison of two runs only means something when both engines saw the same
requests before it. The first prompt a fresh engine sees computes its 16-token
header; every later prompt reuses it, which changes the shape of its first
pass and so its low bits. `st-new-s.json` is the scenario run after six other
prompts: its timings are the second table above, and its outputs are not
comparable with `foff-s.json`.

## One slow start

The first start of this image read about 7% slow on every prefill: RigMark
cold 16K 2,628 tok/s, a lone 19K prompt 2,510 tok/s, the cold-36K prompt's
first token 17.4 s. Decode was normal. It did not come back:

- The 2026-11-14 image, started next on the same hosts: 2,680 tok/s.
- The engine source of this image built with the switch off and on: 2,653
  and 2,690.
- This image started a second time: 2,650, and still 2,680 after replaying
  the whole of the first start's traffic stage by stage; RigMark 2,857.

So neither the image nor its history reproduces it, and the cause is not
found. The speed figures above are from the second start; the identity
checks are from the first, and the suite, greedy and log-probability checks
passed on both. `raw/harness/lone.sh` shows the condition in half a minute
([`raw/yield/lone-console.txt`](raw/yield/lone-console.txt)).

## Limits

- A pass with fewer than 1,024 rows to compute does not pause, nor does the
  first chunk of an image prompt or of a beam search.
- Four Sparks, Strict mode and vision were not run; the switch is not set on
  four Sparks or in Strict mode.
- Not measured on live agent traffic.
- A pause saves and restores the first 128 rows of what a decode step
  overwrites, the widest step the engine takes today. That bound is not
  checked at run time.
- Tool-call grammars and sampled structured output still take a verify pass
  each, as on 2026-11-14.
- Cells are single runs. The exlr8 figures and the figures without the yield
  are from 2026-11-14.
