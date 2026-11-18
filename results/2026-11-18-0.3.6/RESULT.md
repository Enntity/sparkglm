<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.6, 2026-11-18

0.3.6 is 0.3.5 plus one engine fix: a forced tool call can no longer come
back as text. GLM-5.3-Flash FP4-Local's output is unchanged, token for token.
Measured on one DGX Spark pair (TP-2, Spark mode, profile `4x1m`, GPU clock
capped), against 0.3.5's receipts from the same pair.

## The fault

With `tool_choice` set to `required` or to a named tool, the engine masks the
first token of the answer so that it starts a tool call. The mask accepted
the opening tag in two forms: the native `<tool_call>` token, or the same
characters as plain text. GLM's template leaves every prompt inside an open
reasoning block, where both forms are unlikely, and which one ranks higher
depends on the model.

GLM-5.3-Flash FP4-Local ranks the native token first, so nothing showed.
[Sage S2.1 Flash](https://huggingface.co/internlm/Sage-S2.1-Flash), a
checkpoint of the same architecture, ranks the literal `<` first on most
forced-tool prompts. On 0.3.5 such a request then went wrong in three steps:

1. The `<` was emitted as reasoning, which opened `<answer>`, `<content>` or
   a similar invented tag.
2. The reasoning ran unmasked to its end.
3. The grammar resumed in the middle of the tag and forced
   `tool_call>name<arg_key>…` as ordinary text.

The response had no tool call, finished with `stop`, and gave the tool-call
guard nothing to check. One case reasoned to `max_tokens` instead.

## What changed

At the start of a required tool call only the native token is legal
([Enntity/atlas#33](https://github.com/Enntity/atlas/pull/33),
[Atlas-Inf/atlas#395](https://github.com/Atlas-Inf/atlas/pull/395)). Nothing
else changes: `tool_choice: auto`, `response_format` with tools, and every
token after the first are as before.

The engine pin is Enntity/atlas `spark` `01d34b34` (tree `f769ec3c`); the
image is `ghcr.io/enntity/atlas-sparkglm:9d6a18895552`, built with
`ATLAS_IMAGE_CHECKS=0`. It has not been pushed.

## Sage S2.1 Flash

[Enntity/Sage-S2.1-Flash-NVFP4](https://huggingface.co/Enntity/Sage-S2.1-Flash-NVFP4)
at revision `abd1b053`, the live suite of 109 cases:

| | 0.3.5 | 0.3.6 |
|---|---|---|
| Passed | 97 | **109** |
| Failed | 12, all forced-tool guard cases on the Anthropic API | 0 |

The 12 were six cases, each sent blocking and streaming
([0.3.5](raw/sage/suite-summary-0.3.5.json),
[0.3.6](raw/sage/suite-summary-0.3.6.json)). The same cases on the OpenAI API
passed on 0.3.5; the suite sends those with thinking off.

## Unchanged for GLM-5.3-Flash

All against 0.3.5's qualification
([receipts](../2026-11-14-0.3.5/raw/qualification-0.3.5)).

| Check | Result |
|---|---|
| Greedy output, 8 prompts | 8/8 identical |
| Prompt logprobs, 1K / 4.4K / 15K-token prompts | zero difference at 20,705 tokens |
| Live suite and guard log | 109/109, 36 blocks with the expected reasons |
| Prose / code / structured decode, median of 3 | 37.7 / 66.1 / 88.8 tok/s (0.3.5: 37.6 / 65.9 / 87.7) |
| Matrix walls c1-16k / c1-32k / c2-16k / c2-32k / c4-16k | 15.3 / 21.7 / 25.4 / 36.7 / 46.4 s (0.3.5: 15.3 / 21.7 / 25.5 / 36.7 / 46.7) |
| Lone cold 19K-token prefill after the start, 3 prompts | 2,629 / 2,640 / 2,690 tok/s |

Receipts: [`raw/qualification-0.3.6`](raw/qualification-0.3.6).

## Limits

- **Utilization 0.90, not 0.91.** After the image build the head had about
  0.3 GB less memory available than LLooM's admission asks for at the
  recipe's 114 GB, so both ranks ran these checks at GPU utilization 0.90
  under a 113 GB budget. The key-value pool is smaller by that much; nothing
  here fills it. [`raw/harness/pair-mem.py`](raw/harness/pair-mem.py) is the
  edit.
- **One pass.** Speed and matrix rows are one run of the usual harness, not
  three passes.
- **Strict mode with Sage.** Run on the same engine commit before the release
  image was built, Sage in Strict mode passes 97 of 109. The forced-tool
  cases pass; the 12 that fail are three prompts that ask for tool markup
  repeated as plain text, each sent four ways. Sage reasons even when a
  request turns thinking off, Strict mode does not cap it, and the longer
  reasoning leads it to a different answer. Serve Sage in Spark mode.
- **Not run:** Strict mode and image input with GLM-5.3-Flash on this image,
  four Sparks, and an install through LLooM's recipe.
