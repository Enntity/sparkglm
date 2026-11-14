<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.4, 2026-11-14

0.3.4 is 0.3.3 plus one behavior change: in Spark mode a request that turns
thinking off is served at the template's Low effort under a 256-token cap,
instead of reasoning at Max effort with no limit. Requests that reason are
unchanged, bit for bit. Measured on one DGX Spark pair (TP-2, Spark mode,
profile `4x1m`, GPU clock capped), against 0.3.3 on the same pair.

## The fault

GLM's checkpoint template opens a reasoning block whatever a request says, and
treats any effort other than `low` or `high` as Max. A client that sends
`chat_template_kwargs.enable_thinking: false` or `reasoning_effort: "none"`
therefore got Max-effort reasoning with no budget, which is also what the
reference server does. An agent runtime that makes short structured calls with
thinking off (two messages, about 2,500 prompt tokens, a JSON schema,
`max_tokens` 3,072, a 120-second timeout) stalled on 0.3.3: over about 18 hours
none of about 520 such calls closed its reasoning. Each ran to the token limit or
the timeout and was retried.

On a small appraisal prompt with a four-field schema, 0.3.3 took 19–27 s per
thinking-off call and wrote 2,100–4,000 characters of reasoning; one of three
ended at `max_tokens` with no answer.

## What changed

The engine gains `--thinking-off-budget N`
([Enntity/atlas#29](https://github.com/Enntity/atlas/pull/29)): a request whose
thinking resolved off is rendered at Low effort, and its reasoning is closed
after N tokens or half the request's `max_tokens`, whichever is less. It is
unset by default. `install/serve.py` sets 256 in Spark mode and nothing in
Strict mode; `THINKING_OFF=low|reference` chooses either.

The engine pin is Enntity/atlas `spark` `640d7595` (tree `46ccca5c`); the image
is `ghcr.io/enntity/atlas-sparkglm:46fc994aa710`, built with
`ATLAS_IMAGE_CHECKS=0`. It has not been pushed.

## Thinking-off requests

[`raw/thinking-off/verify.py`](raw/thinking-off/verify.py), 15 live checks, all
pass on the final image ([output](raw/thinking-off/verify-final2.out)).

| Request | 0.3.3 | 0.3.4 |
|---|---|---|
| JSON-schema appraisal, thinking off | 19–27 s, 2.1–4.0K characters of reasoning | **2.9–3.2 s**, 22 characters |
| Same, streaming | not measured | 2.3 s |
| Two at once on a 2.5K-token prompt | not measured | 5.4 and 5.6 s |
| Plain question, `max_tokens` 16 | not measured | 0.3 s |
| Tool call with thinking off | not measured | 2.4 s, correct call |
| No thinking flag, or `enable_thinking: true` | the row above: on 0.3.3 it is the same prompt | 18.9 and 27.5 s, 2.6K and 4.0K characters |
| `reasoning_effort: low` | 2.0–3.0 s | 2.3 s |

On the stalled runtime's own traffic after the change, the first 45 requests
all completed with none abandoned by the caller; thinking-off calls reasoned
for 9 to 278 tokens, and two of them reached the cap.

## Unchanged for requests that reason

All against 0.3.3 (image `c5801bd29e45`). These ran on image `e9935dd07f9e`,
built from the same engine tree before the engine change merged; the final
image differs from it in the recorded pin only.

| Check | Result |
|---|---|
| Greedy output, 8 prompts | 8/8 identical |
| Prompt logprobs, 1K / 4.4K / 15K-token prompts | zero difference at every token |
| Live suite and guard log | 109/109 |
| RigMark prose / code / structured decode | 38.8 / 66.8 / 91.1 tok/s (0.3.3: 38.1 / 66.0 / 89.0) |
| RigMark cold 16K prefill | 2,849 tok/s (0.3.3: 2,865) |
| KV pool, two starts | 152,106 and 165,872 blocks (0.3.3: 151,242–166,564) |
| Matrix with thinking on: C1 16K / 32K, C2 16K / 32K, C4 16K | 49.4 / 55.6 / 52.1 / 37.2 / 52.6 tok/s (0.3.3: 49.3 / 55.9 / 52.2 / 37.3 / 52.4) |

RigMark is one round (`40fabca`, 3 decode runs of 4,096 tokens, 6 cold 16K
prefills, `reasoning_effort: low`). Strict mode, vision and TP-4 were not run:
Strict does not pass the new flag, and no other code changed.

**The matrix harness sends thinking-off requests by default.** On 0.3.3 and
earlier those reasoned at Max effort, so its cells measured decode of long
reasoning text. On 0.3.4 in Spark mode they measure short Low-effort answers
and read lower on single streams: 39.6–44.3 / 41.4–44.9 / 46.9–51.3 /
33.6–36.9 / 44.6–47.9 over three runs. Compare default-mode cells only within
0.3.4 and later, or pass `--enable-thinking` to compare with older receipts, as
the row above does.

## Rejected: closing the reasoning block

The first version of the fix ([Enntity/atlas#28](https://github.com/Enntity/atlas/pull/28),
reverted by #29) appended `</think>` to the prompt of a thinking-off request,
so the model answered with no reasoning at all. It was as fast, and greedy
output and logprobs for other requests were identical, but the live suite
scored 107/109 ([receipts](raw/closed-block-attempt)). Both failures were the
literal-tool-XML canary: asked to repeat a tool call as text without invoking
tools, the model called the tool with an invented argument. At Low effort the
same request is answered correctly after 4 reasoning tokens in the same time
([`canary.py`](raw/thinking-off/canary.py)).

## Limits

- A thinking-off prompt in Spark mode is the Low-effort prompt, not the
  reference server's. Strict mode, and `THINKING_OFF=reference`, keep the
  reference rendering.
- A thinking-off answer usually carries a few tokens of `reasoning_content`.
- This does not make structured calls fast. With two of the runtime's agents
  active its requests took a median 18 s and up to 160 s after the change, at
  a median 15 tok/s.
