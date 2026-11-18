<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.6, 2026-11-18: the checks it lacked as a candidate

0.3.6 was published as a candidate on 2026-11-18 after qualifying on a DGX
Spark pair in Spark mode ([results](../2026-11-18-0.3.6/RESULT.md)). It had
not been installed through LLooM's recipe, run in Strict mode, run with images
or video, or run on four Sparks. This is those four checks, all with
GLM-5.3-Flash FP4-Local on the published image
(`ghcr.io/enntity/atlas-sparkglm@sha256:e241d69d…73d8`) pulled by digest.

| Check | Compared with | Result |
|---|---|---|
| Install through LLooM recipe version 7, two Sparks, the recipe's own 0.91 and 114 GB | 0.3.6's qualification run (2026-11-18, utilization 0.90) | Installs and serves; greedy output 8/8 identical; prompt logprobs zero difference |
| Strict mode, two Sparks | 0.3.5 in Strict mode (2026-11-16) | Greedy output 8/8 identical; prompt logprobs zero difference |
| Images and video, two Sparks, Spark mode | 0.3.5 (2026-11-16) | 10/10 identical: text, reasoning and every token logprob |
| Four Sparks (TP-4 chain), Spark mode | 0.3.5 on four Sparks (2026-11-16) | Greedy output 8/8 identical; prompt logprobs zero difference; suite 107/109, the same two cases; speed level |

Prompt logprobs cover 984, 4,429 and 15,292 tokens in each comparison, and the
largest difference in any of them is 0.0000.

## Four Sparks

Installed with recipe version 7 and `--sparks 4` on each Spark of the chain
and started from rank 0 through LLooM.

Decode, median of three runs (`quick_speed.py`), tok/s:

| | 0.3.6 | 0.3.5 |
|---|---|---|
| Prose | 57.5 | 58.3 |
| Code | 106.5 | 106.9 |
| Structured | 146.8 | 146.2 |

The matrix, wall seconds. 0.3.5 was run again on the same four Sparks between
the two 0.3.6 passes, at the same memory setting:

| Case | 0.3.6, two passes | 0.3.5, same day | 0.3.5, 2026-11-16 |
|---|---|---|---|
| One stream, 16K | 10.02 / 10.37 | 10.56 | 9.03 |
| One stream, 32K | 14.73 / 14.09 | 13.66 | 13.00 |
| Two streams, 16K | 19.33 / 17.68 | 16.50 | 17.63 |
| Two streams, 32K | 25.40 / 26.02 | 25.34 | 24.83 |
| Four streams, 16K | 31.03 / 30.19 | 30.25 | 30.40 |

These walls are not a like-for-like speed measure on four Sparks. Each run
writes different text for the same prompt (the same 0.3.5 image wrote 1,395
characters on 2026-11-16 and 1,281 on 2026-11-18 in the first case), and text
that accepts fewer drafted tokens takes more verify steps for the same 400
tokens. The two parts that do not depend on the text are level
([`raw/tp4/matrix-steps.out`](raw/tp4/matrix-steps.out)):

| | 0.3.6, two passes | 0.3.5, same day | 0.3.5, 2026-11-16 |
|---|---|---|---|
| First token, one stream, 16K (s) | 3.87 / 4.00 | 4.08 | 3.81 |
| First token, one stream, 32K (s) | 7.52 / 7.57 | 7.42 | 7.41 |
| Stream-seconds per verify step, all cases | 0.0846 / 0.0845 | 0.0835 | 0.0832 |

On two Sparks the same matrix writes the same text every run, and 0.3.6's
walls there matched 0.3.5's to within 0.3 s
([results](../2026-11-18-0.3.6/RESULT.md)).

## Things to know

- **The four-Spark run used less memory than the recipe sets.** The recipe
  budgets 114 GB a Spark at GPU utilization 0.88. Two Sparks of this chain
  host other services (an embedding model on one; a speech model and a
  transcription model, 30 GB by LLooM's count, on another). All four ranks ran
  at utilization 0.69 under a 90 GB budget. Only the prompt-cache pool is
  smaller: 261,596 blocks (4.2M tokens). A first attempt at 96 GB was refused
  by LLooM's admission, which named the Spark and the shortfall
  ([`raw/tp4/refused-at-96gb.out`](raw/tp4/refused-at-96gb.out)). The recipe's
  own four-Spark setting was not started on this chain.
- **Two suite cases fail on four Sparks:** `anthropic-no-tools-block` and
  `anthropic-no-tools-stream`. Both reason to `max_tokens` with no answer, as
  they did on 0.3.5 and 0.3.0 on four Sparks. They pass on two.
- **One pass of 0.3.5 on the same day, two of 0.3.6.** The per-step figure for
  0.3.6 is 1.3% above 0.3.5's, which is within what one pass can show.
- **Sage S2.1 Flash was not run in these checks.** The reason for 0.3.6 is a
  forced tool call that came back as text with that checkpoint; its 109/109 on
  0.3.6 is in the [qualification results](../2026-11-18-0.3.6/RESULT.md).
- **The image and video run is Spark mode on two Sparks.** Its KV pool read
  155,770 blocks; the Strict start read 169,130.
- **Changing an installed host between two Sparks and four needed no hand
  work this time.** LLooM's setup now replaces the runtimes of the other size.
  A host that had the recipe installed with `--memory shared` kept that option
  until `--memory full` was passed.
- Not run: Strict mode or images on four Sparks, a ring, live agent traffic.

## Receipts

- `raw/install/console.out`: the whole two-Spark run: install on both hosts,
  start, and the three comparisons.
- `raw/spark/`: greedy outputs, logprob dumps and both comparisons of the
  recipe-installed pair against the qualification run.
- `raw/strict/`: the same against the stored 0.3.5 Strict run.
- `raw/vision/`: the image and video run and its comparison.
- `raw/tp4/qualification/`: identity, suite, greedy, logprobs, speed and the
  matrix on four Sparks; `raw/tp4/console.out` has the comparison against the
  stored 0.3.5 run.
- `raw/tp4/matrix-repeat/`: the same-day matrix pass of 0.3.5 and the second
  pass of 0.3.6.
- `raw/tp4/engine-start.log`: start lines of rank 0.
- `raw/harness/`: the scripts that ran the checks. `mode-conf.py`,
  `mem-conf.py` and `vision_eq.py` are those of the
  [0.3.5 promotion](../2026-11-16-0.3.5-promotion/raw/harness).

Host names, user names and addresses in the receipts are replaced with
`spark01` to `spark04`, `user`, `<addr>` and `RANK1_FABRIC_ADDR`.
