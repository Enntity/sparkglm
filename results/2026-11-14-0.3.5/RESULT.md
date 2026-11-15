<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.5, 2026-11-14

0.3.5 is 0.3.4 plus two engine changes to how concurrent streams share a
verify pass. Both leave every token as it was. Measured on one DGX Spark pair
(TP-2, Spark mode, profile `4x1m`, GPU clock capped), against 0.3.4 and
against the exlr8 engine (`kindlingai/glm-5.3-flash-exlr8-2x-spark`, recipe
defaults, variant bit349) on the same pair the same day.

## What was slow

Two agents sharing the pair, each with 25–40K tokens of context and mostly
JSON-schema calls, ran at about 14 tok/s each on 0.3.4. Three separate causes:

1. **Structured output never shared a pass.** A request with a
   `response_format` grammar took a target pass of its own every step. One
   JSON stream decoded at 42 tok/s; two or four decoded at 43–44 combined.
2. **Streams in a shared pass had one width.** Free text did share a pass,
   but every stream verified the same number of drafts, so a prose stream
   paid for rows a code stream wanted. A pass costs about 46 ms plus 5.2 ms a
   row at two streams and 69 ms plus 3.7 ms a row at four, so rows are most
   of it.
3. **A prefill stalls every decoder.** While one request prefills, a decoding
   stream gets one step per 8K-token chunk: 2.9 s gaps, and a 13 s stall
   beside a cold 36K prompt. This is not fixed in 0.3.5 (see Limits).

Context length itself was not a cause: with the prompt cached, 36K-token
contexts decode as fast as 2K ones at one, two and four streams.

## What changed

- **Structured output joins the shared pass**
  ([Enntity/atlas#30](https://github.com/Enntity/atlas/pull/30),
  `ATLAS_GLM_STRICT_OWNER_BATCH`). Each of its rows is picked under its
  grammar mask in the one pass.
- **Each stream verifies its own width**
  ([#31](https://github.com/Enntity/atlas/pull/31),
  `ATLAS_GLM_PAIR_LIVE_WIDTHS`). The rule is priced by step costs fitted to
  4,312 verifies on this pair.

`install/serve.py` sets both in Spark mode on a pair. Strict mode and four
Sparks are unchanged. The engine pin is Enntity/atlas `spark` `225a65eb` (tree
`1344ed41`); the image is `ghcr.io/enntity/atlas-sparkglm:69295e00a748`, built
with `ATLAS_IMAGE_CHECKS=0`. It has not been pushed, and the release pins a
later engine ([2026-11-15 results](../2026-11-15-0.3.5/RESULT.md)).

## Against 0.3.4

Decode tok/s combined, 400 tokens a stream, prompts of about 2,400 tokens
already cached, greedy ([`raw/harness/gbench.py`](raw/harness/gbench.py)).
One run a cell; repeated cells on the same build differed by up to 5%.

| | 1 stream | 2 streams | 4 streams |
|---|---|---|---|
| JSON-schema output, 0.3.4 | 41.8 | 43.7 | 43.1 |
| JSON-schema output, 0.3.5 | 41.8 | **54.7** | **68.5** |
| Free text, 0.3.4 | 42.0 | 53.6 | 64.3 |
| Free text, 0.3.5 | 46.2 | **56.1** | **69.2** |

One stream takes the same path on both versions, so the one-stream column
shows the spread between runs (40.1 to 46.2 over five runs of that free-text
cell), not a gain.

## Against exlr8

The same harness against both engines. "Short" prompts are about 60 tokens,
so the cell is decode alone; "2K" prompts are about 2,400 tokens. End to end
is every token of the wave over its wall time from send to last token. exlr8
refuses `reasoning_effort: "none"`, so its JSON cells ran at low effort;
SparkGLM's ran with thinking off, which 0.3.4 serves at low effort under a
cap.

| Decode tok/s combined, short prompts | 1 stream | 2 streams | 4 streams |
|---|---|---|---|
| Free text, exlr8 | 40.8 | 61.0 | 75.7 |
| Free text, SparkGLM 0.3.5 | **46.1** | 59.7 | 74.6 |
| JSON-schema, exlr8 | 47.5 | 59.8 | 76.9 |
| JSON-schema, SparkGLM 0.3.5 | **53.4** | 61.9 | 77.2 |

| End-to-end tok/s, 2K prompts | 1 stream | 2 streams | 4 streams |
|---|---|---|---|
| Free text, exlr8 | 34.1 | 43.2 | 48.0 |
| Free text, SparkGLM 0.3.5 | **45.3** | **55.4** | **68.7** |
| JSON-schema, exlr8 | 40.7 | 42.3 | 47.6 |
| JSON-schema, SparkGLM 0.3.5 | 41.0 | **53.9** | **67.9** |

| A cold 36K-token prompt arriving beside a decoding stream | exlr8 | SparkGLM 0.3.5 |
|---|---|---|
| Time to first token of the new request | 38.7 s | **13.3 s** |
| End-to-end tok/s of the pair of requests | 22.0 | **33.7** |
| Longest gap in the decoding stream | **0.85 s** | 2.9 s |

Read with the 5% run-to-run spread in mind: at two and four streams on short
prompts the engines are level, SparkGLM leads on one stream, and it leads by
25–45% wherever a 2K prompt is involved, because exlr8 took 1.7 s to its
first token on a 2.4K prompt and started concurrent streams several seconds
apart. On a 2K
prompt exlr8 decodes one JSON stream faster (49.7 against 41.8 tok/s), which
its slower start cancels end to end.

## Same tokens

- [`raw/harness/jexact.py`](raw/harness/jexact.py): 12 JSON-schema prompts and
  12 free-text prompts, each generated alone and then in waves of two, three
  and four with different partners. Every output is identical to its lone
  run (40 of 40 each), and each lone run repeats identically.
- Greedy output on the release set: 8/8 identical to 0.3.4. Prompt logprobs on
  1K, 4.4K and 15K-token prompts: zero difference at every token.
- Live suite and guard log: 109/109. Thinking-off checks from 0.3.4: 15/15.
- [`raw/harness/soak.py`](raw/harness/soak.py), 180 s of mixed traffic at four
  streams (strict JSON, streaming JSON, sampled JSON, `json_object`, forced
  tool calls, free text): no engine errors. 66 of 67 replies valid; the other
  was a free-text request that spent its 250-token limit reasoning.
- RigMark, one round: prose 38.7, code 67.1, structured 90.9 tok/s, cold 16K
  prefill 2,851 (0.3.4: 38.8, 66.8, 91.1, 2,849).

## Limits

- **Prefill still stalls decoders** in the build measured here. The release
  fixes it another way, by letting a prefill pass yield between layers:
  [2026-11-15 results](../2026-11-15-0.3.5/RESULT.md). The engine's ordering switch
  (`ATLAS_DECODE_STEPS_PER_CHUNK`) gives decoders steps between chunks but
  cannot shorten the 2.9 s a chunk takes, and it lengthened the new
  request's wait by 2.7 s in one trial. A shorter gap needs smaller chunks
  under load, which changes prefill numerics.
- **Tool-call grammars and sampled structured output** still take a pass
  each.
- **exlr8 is level, not beaten, on pure decode at two and four streams.**
- Not run: Strict mode, vision, TP-4. Neither switch is set there.
- Cells are single runs; the three builds measured during the work are in
  [`raw/cells`](raw/cells).
