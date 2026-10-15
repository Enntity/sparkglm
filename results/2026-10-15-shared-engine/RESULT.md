# 2026-10-15: SparkGLM on the shared Spark engine (sparkglm/atlas-20261015-vision)

The engine is now `sparkglm/atlas-20261015-vision` at `16315004` (tree `e356f199`): Atlas-Inf `main` at `7ffd183f`,
the shared layer `upstream/atlas-shared-20261015-vision` at `a9464810` that SparkQwen builds on too, and the
SparkGLM-only commits. The image tag is `f0f8a55c3da2`. It replaces production's `2b2970c2f05d` (engine
`sparkglm/atlas-20261014-vpc` at `289cefbc`). The shared layer is the merged layer `fb80cbe0`, compared against
production below, plus five vision fixes both products share ([Vision fixes](#vision-fixes)). Text serving is meant to
be unchanged: the same prompts get the same prompt logprobs on every engine here.

## What changed

- **One engine for both products.** SparkGLM and SparkQwen had separate engine branches (`289cefbc` here,
  `sparkqwen/atlas-20261013-prefill` there), each with its own copy of the GLM port, the RC4 commits and the
  model-independent serving work. Both now sit on one layer; each product adds only its own commits
  ([`install/atlas-source.json`](../../install/atlas-source.json) records the layers).
- **SparkGLM now runs SparkQwen's model-independent RC16 commits:** speculation commits exactly the tokens serial
  decode commits, one lowest-index rule for exact greedy ties, shared-prefix admission, opt-in loop steering and
  prefill burst (both off in SparkGLM's profiles), and the step trace.
- **Three fixes the merge needed, all on the shared layer:**
  - The `w4a16_gemv_qg_batch4` kernel lookup is gated to `qwen4_exp`. Without the gate GLM's startup kernel audit
    refused to start.
  - Sequence-parallel prefill stays on with the DFlash drafter when the drafter's context window fits rank 0's half
    of the rows. SparkQwen's guard turned it off whenever DFlash was on, and GLM's cold prefill got about 14% slower
    (`raw/prefill-pfnew.json`: 45.5 s at 104K tokens against 39.8 s; the four-session fill took 373–377 s against 328–331 s).
  - `ATLAS_MOE_INPLACE_TRANSPOSE=0` selects the consolidated MoE transpose. The SparkGLM profiles set it. The
    in-place transpose, the shared default that SparkQwen relies on, keeps every expert's allocation alive. On GB10
    that cost rank 0 about 1.1 GB of pre-KV memory, mostly kernel slab, and about 14% of the KV pool
    (`raw/kv-newf*.txt`: 103.59 GB pre-KV, 126,726 blocks).
- The plain-decode GPU argmax paths defer to the host pipeline when loop steering may act on a span. This keeps the
  guard on every decode path; it is a no-op while steering is off.

## Checks (ennspark01/02, production settings)

These ran on the merged layer before the vision fixes (`93b3e190`, tree `8b8b04db`). The `4x1m` profile, 0.91, display carveout, KV shard, NVMe prefix cache. Production and the new engine ran back to
back on the same pair, each from a fresh start with an empty NVMe cache directory (`raw/ab_up.sh`). The new engine
ran as a locally built image from the same engine commit, `ee487c14741a` (`93b3e190`). The release image's smoke run
is in [Release image](#release-image).

**Text output unchanged.** The prompt-logprob hash of the fixed 48,875-token text is `246cd1331b41` on both, twice
each, which is RC4.2's published hash (`raw/lp-glm-*.json`).

| | New engine | Production |
|---|---:|---:|
| Prefix-cache test (45K multi-turn + replay) | 8/8 | 8/8 |
| Long tool call (one `write_file`, 3,516 tokens) | intact | intact |
| Loop-repeat repro (`same_cmd_again`) | false | false |
| Decode, 12 prompts × 256 forced tokens, prose / code, tok/s | 44.18 / 63.35 | 39.98 / 61.22 |
| Mean tokens per step, prose / code | 2.51 / 4.90 | 2.51 / 4.87 |
| Cold prefill TTFT at 6.4K / 26K / 52K / 101–106K tokens (s, mean of two unique prompts each) | 2.90 / 10.80 / 20.44 / 39.22 | 2.87 / 10.60 / 20.39 / 39.84 |
| Fill: four concurrent ~204K sessions × 3 turns, correct | 12/12 in 329.0 s | 12/12 in 327.9–331.3 s (three starts) |
| Lowest rank-0 MemAvailable during the fill (`raw/mem-fill-*.log`) | 4.59 GB | 4.01–4.61 GB |
| Rank 0 before the KV pool (free-memory measure) | 102.49 GB | 102.42–102.45 GB |
| KV pool, carveout included (`raw/kv-*.txt`) | 145,592 blocks | 146,548–147,336 blocks |

The pool size comes from a free-memory measurement that moves by about 1% between starts of the same image. The
tracked device footprint is 99.36 GB on the new engine against 99.30 GB on production. The new engine's decode is no
slower: the tokens per step are the same, and the production arm's prose rate here is lower than its published
figure (44.1 tok/s on 2026-10-14), so the 10% prose difference is run-to-run spread rather than a speedup.

**Images** (`raw/vpc-glm/`). The new engine matches production:
- **Interleaved race.** A blue square at the end of 47,772 tokens and a green triangle sent 3 s later: A answers
  "Blue square", B "green triangle".
- **68K-token conversation with one image.** Turn 1 is cold at 26.5 s. Turns 2–5 start in 0.71–1.00 s with the image
  prefix cached.
- **Swapped order, a different image at the same position, a re-sent image:** all answered correctly, and only
  identical content is reused.

## Release candidate without the vision fixes

`948aabe47fa4` (engine `93b3e190`) was built on ennspark01 by `install/build.sh` from the published pin (a fresh clone of this branch), all
14 build-time check stages passed, and the same image was loaded on ennspark02. It then ran the checks above on the
pair (`raw/release/`, log `rel-smoke.out`):

- Prompt-logprob hash `246cd1331b41`, twice; prefix-cache test 8/8; long tool call intact; loop-repeat repro false.
- Decode prose / code 44.38 / 63.66 tok/s (2.51 / 4.90 tokens per step).
- Cold prefill 2.87 / 10.84 / 20.49 / 40.01 s at 6.4K / 26K / 52K / 105K tokens.
- Images: race correct, 68K conversation turns 2–5 at 0.72–1.00 s, swap and re-send answers identical to the table's
  runs.
- Fill 12/12 in 327.1 s; lowest rank-0 MemAvailable 4.55 GB; 102.46 GB before the KV pool, 146,246 blocks.
- No panics or errors on either rank.

## Vision fixes

Five commits on the shared layer (`fb80cbe0..a9464810`), found on SparkQwen, whose image prompts could crash the pair
or read another request's image. They change no text path:

- Every vision tower encodes per chunk into per-sequence state (GLM-5 already did).
- A malformed vision layout (placeholders the encoded rows do not cover) fails only its request. Every rank decides it
  from the same data, so the EP worker keeps serving instead of ending the pair.
- MRoPE positions for the generic walker come from the whole prompt, as GLM-5's already did.
- Vision markup typed in a media request's message text (`<|image_pad|>` in pasted code) encodes as text; only the
  template's own markers are placeholders. `ATLAS_VISION_MARKUP_AS_TEXT=0` turns it off.
- Every tower's image prompts are keyed for the prefix cache (GLM-5's already were).

## Release image

`f0f8a55c3da2` (engine `16315004`) was built on ennspark01 by `install/build.sh` from the published pin; all 14
build-time check stages passed and the image was loaded on ennspark02. On the pair (`raw/vision/`, log
`vis-run.out`):

- Prompt-logprob hash `246cd1331b41`, twice; prefix-cache test 8/8; long tool call intact; loop-repeat repro false.
- Images: the race, the 68K-token conversation and the swap/re-send checks give text byte-identical to the engine
  without the vision fixes (`raw/vision/vpc-glm/`); conversation turns 2–5 start in 0.71–0.99 s from the cache.
- Decode prose / code 43.96 / 63.10 tok/s (2.51 / 4.90 tokens per step).
- Cold prefill 2.93 / 10.79 / 20.15 / 39.92 s at 6.4K / 26K / 51K / 102–106K tokens.
- Fill 12/12 in 329.6 s; lowest rank-0 MemAvailable 4.15 GB; 102.50 GB before the KV pool, 145,379 blocks.
- No panics or errors on either rank.

## RigMark

[RigMark](https://github.com/alexellis/rigmark) (Alex Ellis, MIT) at `40fabcaf`, the release-suite cells and command
lines of [2026-10-12-rc4](../2026-10-12-rc4/RESULT.md) (`rig_glm` in `raw/vision/vis_run.sh`), thinking on with low
effort, against the server directly. Since 2026-10-13 ennspark01/02 run with their GPU clocks capped
(`gpu-clock-cap.service`, `nvidia-smi -lgc 300,2200`, after ennspark02's power-offs); the README's figures (RC4) were
measured before that. The headline run therefore lifted the cap for its duration (`CLOCKS=uncapped` in
`raw/vision-uncapped/vis_rig.sh`: the service is stopped on both Sparks, which resets the clocks, and started again
when the run ends; the GPUs ran at 2,437–2,522 MHz under load). Production serving stays capped.

| | This release, uncapped | README: RC4, uncapped (2026-10-12) | This release, capped | Current production, capped (2026-10-14, 03/04) |
|---|---:|---:|---:|---:|
| Output gates, decode run / concurrency run | 9/9, 3/3 | not in the card, 3/3 | 9/9, 3/3 | 9/9, 3/3 |
| Decode, code / prose / structured, tok/s (median of 3) | 69.1 / 39.3 / 91.9 | 72.3 / 38.3 / 91.7 | 68.9 / 38.6 / 92.0 | 66.7 / 37.2 / 90.4 |
| Cold prefill 8K / 32K / 64K, TTFT (s) | 3.29 / 12.48 / 23.83 | 3.29 / 12.47 / 23.91 | 3.39 / 12.84 / 24.37 | 3.63 / 13.35 / 25.51 |
| Staggered arrivals, prefill first: newcomer TTFT at 2 / 4 (s) | 2.81 / 3.54 | 2.80 / 3.54 | 2.93 / 3.70 | 3.05 / 4.38 |
| Short code, 1 / 2 / 3 / 4 streams aggregate, tok/s | 47.7 / 58.8 / 62.7 / 72.8 | 47.0 / 60.5 / 66.8 / 70.9 | 48.4 / 56.4 / 63.4 / 71.3 | 48.2 / 58.3 / 63.3 / 68.7 |

Uncapped, prefill and staggered arrivals match RC4 to within 0.3%. RigMark samples, so each decode run writes a
different program: this release's three code runs wrote 2,454 / 2,096 / 2,588 tokens at 67.5 / 69.1 / 70.5 tok/s,
RC4's 2,172 / 2,360 / 1,813 tokens at 68.8 / 72.3 / 72.4. The equal-text probe (same prompts, forced length) shows the
same tokens per step as production and no slower steps ([Checks](#checks-ennspark0102-production-settings)), so the
code row's 4% is within RigMark's run-to-run spread rather than a decode change. Capped, the release is ahead of
current production's capped run on every row but two-stream short code.

## Receipts

`raw/` holds every receipt: the harness (`ab_up.sh`, `glm_suite.sh`, `glm_final.sh`, `fill_only.sh`,
`prefill_probe.py`, `measure.sh`, `pf_var.sh`), logs (`ab-run2.log` for production, `glm-final.out` for the new
engine) and JSON results. Files named `*prod*` are production, `*gfinal*` the final engine, and `*new*`/`*fnew*`/
`*pfnew*` the intermediate builds before the transpose and sequence-parallel fixes, kept because the text above
cites them. `raw/release/` holds the first release candidate's build log, scripts and smoke receipts (its `rel-smoke.out`
also covers SparkQwen's). `raw/vision/` holds the release image's: build, suite, fill and RigMark receipts
(`rig-gv-*`, capped, with the appliance metadata it declared); `raw/vision-uncapped/` holds the uncapped
RigMark run (`rig-gvu-*`).
