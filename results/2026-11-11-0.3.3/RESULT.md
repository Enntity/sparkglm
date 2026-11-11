<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.3.3, 2026-11-11

0.3.3 is 0.3.2 plus three engine changes that leave output bit-identical, with
the reference DFlash2 drafter at its Apache-2.0 revision. Measured on one DGX
Spark pair (TP-2, Spark mode, profile `4x1m`, GPU clock capped), against 0.3.2
on the same pair.

| | 0.3.2 | 0.3.3 |
|---|---|---|
| RigMark prose decode | 34.7 tok/s | **38.1** (+9.7%) |
| RigMark code decode | 62.6 | **66.0** (+5.4%) |
| RigMark structured decode | 81.8 | **89.0** (+8.8%) |
| RigMark cold 16K prefill | 2,753 tok/s | **2,865** (+4.1%) |
| KV pool (0.91, display carveout) | 136,362–142,448 blocks, 2.18–2.28M tokens | **151,242–166,564 blocks, 2.42–2.67M tokens** (two starts) |
| Matrix C1 16K / 32K, decode | 41.2 / 43.0 tok/s | **49.3 / 55.9** |
| Matrix C2 16K / 32K, aggregate | 44.5 / 32.8 | **52.2 / 37.3** |
| Matrix C4 16K, aggregate | 45.1 | **52.4** |

RigMark: `40fabca`, one round each, 3 decode runs of 4,096 tokens and 6 cold
16K prefills, `reasoning_effort: low`. The 0.3.2 round ran with the drafter its
LLooM recipe installs on two Sparks (Enntity's v81), so the decode rows compare
both the engine and the drafter. The matrix is one sample a cell and its C4
cell has swung ±8% between runs before.

## What changed

1. **A pair under expert TP may verify one draft.** A lone sequence's 2-row
   verify was priced 10.3 ms above its 3-row one, a 2026-09-28 measurement of a
   path expert TP retired. It measures 42.0 ms against 47.4. About 45% of
   prose steps accept no draft, so the narrow step pays: prose +5.8% with
   0.3.2's drafter. ([Enntity/atlas#25](https://github.com/Enntity/atlas/pull/25))
2. **Quantized MoE and MLA gathers on the pair** in Spark mode. During
   sequence-parallel prefill each Spark sends the packed form the other would
   compute anyway; a fifth fewer bytes cross the link.
   ([#26](https://github.com/Enntity/atlas/pull/26))
3. **Three per-pass buffers sized for GLM-5.** The prefill arena is allocated
   before the KV pool, so what it gives back goes to the prompt cache.
   ([#27](https://github.com/Enntity/atlas/pull/27))
4. **The drafter pin moves to `6cc6c481`**, the revision its publisher released
   under Apache-2.0 on 2026-11-10. It carries the 2026-08-31 weights, not the
   2026-08-27 ones that `7d74cdd8` pinned. On the same engine the two measured
   level: prose 37.7 against 38.3, code 66.3 against 65.6, structured 87.9
   against 89.3.

The engine pin is Enntity/atlas `spark` `a703c117` (tree `490d70db`); the image
is `ghcr.io/enntity/atlas-sparkglm:c5801bd29e45` (install tree `c5801bd2`, image
ID `sha256:b37843fa…`), built with `ATLAS_IMAGE_CHECKS=0`.

## Checks

All against 0.3.2 (image `fd4bdf836fbe`) on the same pair.

| Check | Result |
|---|---|
| Greedy output, 8 prompts, Spark | 8/8 identical |
| Prompt logprobs, 1K / 4.4K / 15K-token prompts, Spark | zero difference at every token |
| Greedy output and prompt logprobs, Strict | 8/8 identical, zero difference |
| Image and video output (10 cases, to a 19K-token video, alone and beside text decodes) | 10/10 identical, every token logprob |
| Live suite and guard log | 109/109 |
| Engine unit tests for the three changes | pass |

The drafter only proposes tokens and the target verifies each one, so the
drafter revision changes speed, not output: the Spark rows above compare
0.3.3 with the new revision against 0.3.2 with v81.

## Not covered

- Four Sparks (TP-4). The 2-row price and the quantized gathers are pair-only;
  the buffer sizes apply to every topology and were run on the pair only.
- The disk prefix-cache tier and long evicted conversations were not re-run.
- Speed is one RigMark round and one qualification pass per build.
- The larger 16K prefill chunk schedule (about +7% prefill) is not used: it
  costs more than half the KV pool.

## Receipts

`raw/rigmark-0.3.3`, `raw/rigmark-0.3.2`: RigMark JSON, card and container
environment. `raw/qualification-0.3.3`: identity, suite, guard log, greedy
texts, logprob dump, speed and matrix. `raw/baseline-0.3.2`: the 2026-11-10
0.3.2 run the comparisons use. `raw/strict`, `raw/vision`: the Strict and
image/video comparisons. `raw/build-and-run.log`: the build and boot log.

```sh
sha256sum -c SHA256SUMS
```
