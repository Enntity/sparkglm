<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.5, 2026-11-15: prompts prefill in arrival order

The [2026-11-14 results](../2026-11-14-0.3.5/RESULT.md) left prefill as the
remaining cause of slow concurrent streams. This page covers two changes to
how prefill is scheduled, both of which leave every token as it was. One is
in the release: a prompt that arrives while another is prefilling waits its
turn. The other, a prefill pass that yields to decoding streams between
layers, was built and measured and is left off, because it costs wall time
under load. Measured on one DGX Spark pair (TP-2, Spark mode, profile `4x1m`,
GPU clock capped).

## What the release sets

`install/serve.py` sets `ATLAS_PREFILL_QUEUE_FIRST_CHUNK=1` in Spark mode on a
pair. The switch was already in the engine and SparkGLM had not set it. It no
longer sets `ATLAS_GLM_PREFILL_YIELD_MS`
([Enntity/atlas#32](https://github.com/Enntity/atlas/pull/32)); a profile's
environment can. Strict mode and four Sparks are unchanged.

The engine pin is Enntity/atlas `spark` `59ead299` (tree `dfd866dc`), the same
engine as the yield build below. The image is
`ghcr.io/enntity/atlas-sparkglm:503ab09654f0`, built with
`ATLAS_IMAGE_CHECKS=0`. It has not been pushed.

## Arrival order

Every new prompt ran its first 8K-token chunk when it was admitted, ahead of
the chunks an earlier prompt still had to run. So the first of several
prompts arriving together waited for a chunk of each of the others before its
own second chunk. With the switch, a prompt longer than one chunk that
arrives while another is prefilling joins the queue without running anything.
Each prompt's chunks are what they were; only their order across prompts
changes.

Four cold prompts one second apart, 400 output tokens each, median of three
runs ([`raw/harness/field2.sh`](raw/harness/field2.sh); captures in
[`raw/release/waves`](raw/release/waves)). "Before" is the 2026-11-14 image,
"yield" is the build of the next section, and exlr8 is
`kindlingai/glm-5.3-flash-exlr8-2x-spark:latest` with `TUNE=speed` on the same
pair and the same prompts:

| 16K-token prompts | This release | Before | Yield at 300 ms | exlr8 |
|---|---|---|---|---|
| First token, stream 1 | **5.9 s** | 11.2 s | 11.3 s | 9.4 s |
| First token, streams 2 / 3 / 4 | 17.9 / 24.0 / 11.8 s | 20.4 / 23.6 / 17.1 s | 15.2 / 19.4 / 27.8 s | 43.0 / 59.9 / 62.1 s |
| Mean first token | **14.9 s** | 18.1 s | 18.4 s | 43.6 s |
| Whole wave | 46.2 s | **45.7 s** | 47.7 s | 75.4 s |
| Longest stall of a stream after its first token | 3.2 s | 5.9 s | **0.5 s** | 2.3 s |

| 32K-token prompts | This release | Before | Yield at 300 ms | exlr8 |
|---|---|---|---|---|
| First token, stream 1 | **11.8 s** | 20.0 s | 20.0 s | 24.9 s |
| First token, streams 2 / 3 / 4 | 36.3 / 48.8 / 24.0 s | 38.8 / 48.3 / 29.3 s | 43.8 / 57.3 / 31.4 s | 68.2 / 112.3 / 93.8 s |
| Mean first token | **30.2 s** | 34.1 s | 38.1 s | 74.8 s |
| Whole wave | 70.6 s | **70.2 s** | 73.6 s | 122.6 s |
| Longest stall of a stream after its first token | 3.0 s | 3.0 s | **0.75 s** | 1.6 s |

The first prompt gets its first token in about the time it takes alone
(5.7 s for 16K), and the wave is as long as before within the spread of its
three runs (46.0 to 46.7 s against 45.6 to 46.4 s).

Stream 4 is served second, ahead of streams 2 and 3. Its prompt is one token
shorter than theirs (16,191 against 16,192), and the profile's
shortest-prompt-first rule (`ATLAS_PREFILL_SRPT`, in the profiles since 2026-09-30) lets
a shorter prompt overtake. The engine log shows it queued for 2 ms where the
other two waited 9.1 s and 15.1 s
([`engine-prefill.log`](raw/release/waves/release/engine-prefill.log)). The
mean is the same in either order. The same rule ordered the wave before this
change.

The matrix, wall seconds, mean of three passes with the same prompts on each
build ([`bench/run_matrix.py`](../../bench/run_matrix.py), salts
`mb-1115-p1` to `p3`); exlr8 ran the first pass only:

| Case | This release | Before | Yield at 300 ms | exlr8, pass 1 |
|---|---|---|---|---|
| 1 stream, 16K | 13.6 | 13.6 | 13.6 | 17.3 |
| 1 stream, 32K | 19.2 | 19.2 | 19.2 | 28.1 |
| 2 streams, 16K | 25.5 | 25.2 | 25.5 | 37.1 |
| 2 streams, 32K | 36.6 | 36.5 | 37.1 | 57.7 |
| 4 streams, 16K | 44.9 | 44.8 | 46.6 | 72.3 |
| Sum | 139.8 | 139.4 | 141.9 | 212.5 |

On the first pass alone the three SparkGLM builds sum to 138.2, 138.4 and
140.1 s.

The matrix reports a lower decode rate for the release on the multi-stream
cases (four streams: 40 tok/s against 47 before). That figure counts from
each stream's first token, which now comes earlier while the stream still
waits on the other prompts. The wall time is the comparison.

### Same tokens

- **Against the build before it, in the waves.** Every stream of every
  capture above produced the same text on this release, on the 2026-11-14
  image and on the yield build: 32 streams of 400 tokens each, greedy
  (`python3 raw/harness/sametext.py raw/release/waves before release`).
- **Alone against waves.** 12 JSON-schema and 12 free-text prompts, each
  generated alone and in waves of two, three and four: 40 of 40 identical
  each.
- **Against 0.3.4.** Greedy output 8/8 identical; prompt log-probabilities on
  1K, 4.4K and 15K-token prompts zero difference at every token. Live suite
  and guard log 109/109. Thinking-off checks 15/15.
- **Soak**, 180 s of mixed traffic at four streams: no engine errors. 65 of
  66 replies valid; the other was a free-text request that spent its
  250-token limit reasoning, as on 2026-11-14.

Receipts: [`raw/release`](raw/release) (`final.out` is the console of
[`raw/harness/final035g.sh`](raw/harness/final035g.sh)).

### Speed elsewhere

RigMark, one round: prose 38.7, code 67.4, structured 91.2 tok/s, cold 16K
prefill 2,856 (2026-11-14: 38.7, 67.1, 90.9, 2,851). Decode with prompts
already cached, tok/s combined at 1 / 2 / 4 streams on 2.4K-token prompts,
one run a cell: free text 40.2 / 56.4 / 71.0, JSON-schema 48.5 / 56.2 / 68.9
(2026-11-14: 46.2 / 56.1 / 69.2 and 41.8 / 54.7 / 68.5). The one-stream cells
moved in opposite directions by more than the others; they are single runs
and nothing in this change touches a lone stream.

## The layer yield, measured and left off

A prefill pass can stop between two layers, let the scheduler run its
ordinary decode step, and continue
([Enntity/atlas#32](https://github.com/Enntity/atlas/pull/32),
`ATLAS_GLM_PREFILL_YIELD_MS`). The pass computes the same layers over the
same rows in the same order, so its output is bit for bit what it was. The
figures in this section are from an image that set a 300 ms target in Spark
mode on a pair, `ghcr.io/enntity/atlas-sparkglm:712cf8246f85`, on the same
engine pin.

It does what it was built for: a decoding stream's longest freeze goes from
about 3 s to about 0.5 s. It is not in the release because nothing else
improves and waves take longer: 1.8% on the matrix above and 4 to 5% on the
four-prompt waves. A stream that has been decoding for a second or two is
already far ahead of anyone reading or listening to it, so the freeze it
removes is mostly not seen. To turn it on, add
`"ATLAS_GLM_PREFILL_YIELD_MS": "300"` to a profile's `environment`.

### What it buys and what it costs

A cold 36K-token prompt arriving beside one decoding stream
([`gbench.py`](../2026-11-14-0.3.5/raw/harness/gbench.py), one run each):

| | 0.3.5 without the yield (11-14) | Yield at 300 ms | exlr8 (11-14) |
|---|---|---|---|
| Longest gap in the decoding stream | 2.93 s | **0.52 s** | 0.85 s |
| Time to first token of the new request | **13.2 s** | 16.4 s | 38.7 s |
| End-to-end tok/s of the pair of requests | **33.7** | 33.3 | 22.0 |

Two decoding streams while five prompts arrive
([`raw/harness/yexact.py`](raw/harness/yexact.py)); the same engine source
with the switch off (2026-11-14) against the yield image:

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

| Staggered cold prompts, wall time of the wave | 0.3.5 without the yield | Yield at 300 ms |
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
(not run on that image).

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
| Free text, yield at 300 ms | 43.2 | 53.7 | 69.8 |
| JSON-schema, without the yield | 41.8 | 54.7 | 68.5 |
| JSON-schema, yield at 300 ms | 42.3 | 56.1 | 74.5 |

Two free-text streams read 53.7 and 53.8 on the two starts of the yield
image, against 54.8 to 56.4 over the four earlier runs and 56.4 on the
release. That may be a small cost of the yield and is not resolved.

### Same tokens with the yield on

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

The first start of the yield image read about 7% slow on every prefill: RigMark
cold 16K 2,628 tok/s, a lone 19K prompt 2,510 tok/s, the cold-36K prompt's
first token 17.4 s. Decode was normal. It did not come back:

- The 2026-11-14 image, started next on the same hosts: 2,680 tok/s.
- The engine source of that image built with the switch off and on: 2,653
  and 2,690.
- The yield image started a second time: 2,650, and still 2,680 after replaying
  the whole of the first start's traffic stage by stage; RigMark 2,857.

So neither the image nor its history reproduces it, and the cause is not
found. The speed figures above are from the second start; the identity
checks are from the first, and the suite, greedy and log-probability checks
passed on both. `raw/harness/lone.sh` shows the condition in half a minute
([`raw/yield/lone-console.txt`](raw/yield/lone-console.txt)).

Five later starts on 2026-11-15, of the 2026-11-14 image, the yield image and
two builds of the release, each read 2,610 to 2,690 tok/s on that check
(`raw/release/final.out` and the consoles in `raw/release/waves`).

## Limits

- Arrival order is set and measured on the pair only. A prompt that fits one
  chunk, an image prompt and a beam search still run their first chunk when
  admitted.
- The queue is ordered by prompt length, not by rows left to compute, so a
  long prompt that is mostly cached does not go ahead of a shorter cold one.
- Four Sparks, Strict mode and vision were not run; neither switch is set on
  four Sparks or in Strict mode.
- Not measured on live agent traffic.
- With the yield on, a pass with fewer than 1,024 rows to compute does not
  pause, nor does the first chunk of an image prompt or of a beam search. A
  pause saves and restores the first 128 rows of what a decode step
  overwrites, the widest step the engine takes today; that bound is not
  checked at run time.
- Tool-call grammars and sampled structured output still take a verify pass
  each, as on 2026-11-14.
- Cells are single runs. In the yield section, the exlr8 figures and the
  figures without the yield are from 2026-11-14.
- `bench/run_matrix.py field` sent its salt as `cache_salt` until this
  release, which the engine no longer keys its prefix cache on, so its runs
  after the warmup were warm waves (23.8 s). The waves here carry the salt in
  the prompt, and the script now does too.
