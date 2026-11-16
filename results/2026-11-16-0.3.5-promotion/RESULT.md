<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.5, 2026-11-16: the checks it lacked as a candidate

0.3.5 was published as a candidate on 2026-11-15 after qualifying on a DGX
Spark pair in Spark mode ([2026-11-14](../2026-11-14-0.3.5/RESULT.md),
[2026-11-15](../2026-11-15-0.3.5/RESULT.md)). It had not been installed through
LLooM's recipe, run in Strict mode, run with images or video, or run on four
Sparks. This is those four checks, all on the published image
(`ghcr.io/enntity/atlas-sparkglm@sha256:ab56ad3e…b872`) pulled by digest.

| Check | Compared with | Result |
|---|---|---|
| Install through LLooM recipe version 6, two Sparks | — | Installs and serves; ready 53 s after start; thinking-off checks 15/15 |
| Strict mode, two Sparks | 0.3.3 in Strict mode (2026-11-11) | Greedy output 8/8 identical; prompt logprobs zero difference |
| Images and video, two Sparks, Spark mode | 0.3.3 (2026-11-11) | 10/10 identical: text, reasoning and every token logprob |
| Four Sparks (TP-4 chain), Spark mode | 0.3.0 on four Sparks (2026-11-03) | Greedy output 8/8 identical; prompt logprobs zero difference; suite 107/109 (0.3.0: 105/109); speed level |

Prompt logprobs cover 984, 4,429 and 15,292 tokens in each comparison, and the
largest difference in any of them is 0.0000.

## Four Sparks

Installed with recipe version 6 and `--sparks 4` on each Spark of the chain
and started from rank 0 through LLooM. Ready 48 s after start.

Decode, median of three runs (`quick_speed.py`), tok/s:

| | 0.3.5 | 0.3.0 |
|---|---|---|
| Prose | 58.3 | 59.3 |
| Code | 106.9 | 105.2 |
| Structured | 146.2 | 144.3 |

The matrix, wall seconds, three cold passes of 0.3.5 with thinking on against
0.3.0's one pass:

| Case | 0.3.5 | 0.3.0 |
|---|---|---|
| One stream, 16K | 8.86 / 8.12 / 8.54 | 8.89 |
| One stream, 32K | 11.80 / 12.70 / 12.93 | 12.74 |
| Two streams, 16K | 15.22 / 14.99 / 14.65 | 15.53 |
| Two streams, 32K | 22.18 / 22.14 / 22.61 | 22.79 |
| Four streams, 16K | 27.61 / 28.04 / 26.42 | 28.25 |

None of 0.3.5's three switches is set on four Sparks, so this is the 0.3.4
behavior on the 0.3.5 engine, as intended.

## Things to know

- **The four-Spark run used less memory than the recipe sets.** The recipe
  budgets 114 GB a Spark at GPU utilization 0.88. The Spark at rank 2 hosts
  other services and had 111 GB available, so LLooM refused the start. All four
  ranks ran at utilization 0.82 under a 106 GB budget. Only the prompt-cache
  pool is smaller: 426,500 blocks (6.8M tokens). The recipe's own setting was
  not started on this chain.
- **Two suite cases fail on four Sparks:** `anthropic-no-tools-block` and
  `anthropic-no-tools-stream`. Both reason to `max_tokens` with no answer. They
  failed the same way on 0.3.0 on four Sparks and pass on two. The two OpenAI
  cases that failed beside them on 0.3.0 pass now, through 0.3.4's handling of
  requests that turn thinking off.
- **The qualification's own matrix reads slower and is not like for like.** It
  sends thinking-off requests, which 0.3.0 reasoned through at full effort and
  0.3.4 and later answer briefly. Its two- and four-stream cases read 16.85,
  23.52 and 29.18 s against 0.3.0's 15.53, 22.79 and 28.25. The table above
  repeats the matrix with thinking on.
- **The drafter differs from 0.3.0's run.** 0.3.5 pins the reference drafter
  at `6cc6c481`; the 0.3.0 run used its earlier revision. Greedy output and
  logprobs do not depend on the drafter.
- **Strict is compared with 0.3.3**, the last version run in Strict mode.
  0.3.4 sets nothing in Strict mode and was not run there.
- **The image and video run is Spark mode on two Sparks.** Its KV pool read
  152,932 blocks; the Strict start read 169,130. The pool varies by start as on
  0.3.3.
- **Changing an installed host between two Sparks and four needed hand work in
  LLooM's config.** That belongs to LLooM's setup, not to this image, and the
  current default recipe behaves the same.
- Not run: Strict mode or images on four Sparks, a ring, live agent traffic.

## Receipts

- `raw/strict/`: greedy outputs, logprob dumps and both comparisons against
  the stored 0.3.3 Strict run.
- `raw/vision/`: the image and video run and its comparison.
- `raw/tp4/qualification/`: identity, suite, greedy, logprobs, speed and the
  default matrix on four Sparks; `raw/tp4/console.out` has the comparison
  against the stored 0.3.0 run.
- `raw/tp4/matrix-thinking-on/`: the three like-for-like matrix passes.
- `raw/tp4/engine-start.log`: the start lines of rank 0, addresses removed.
- `raw/harness/`: the scripts that ran the checks. A rank's fabric address is
  replaced with `RANK1_FABRIC_ADDR`.
