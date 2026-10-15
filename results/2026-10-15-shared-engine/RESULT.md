# 2026-10-15: SparkGLM on the shared Spark engine (sparkglm/atlas-20261015-shared)

The engine is now `sparkglm/atlas-20261015-shared` at `93b3e190` (tree `8b8b04db`): Atlas-Inf `main` at `7ffd183f`,
the shared layer `upstream/atlas-shared-20261015` at `fb80cbe0` that SparkQwen builds on too, and the SparkGLM-only
commits. The image tag is `948aabe47fa4`. It replaces production's `2b2970c2f05d` (engine
`sparkglm/atlas-20261014-vpc` at `289cefbc`). Nothing in serving is meant to change: the same prompts get the same
prompt logprobs, and the checks below compare the two engines on the production pair.

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

The `4x1m` profile, 0.91, display carveout, KV shard, NVMe prefix cache. Production and the new engine ran back to
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

## Release image

`948aabe47fa4` was built on ennspark01 by `install/build.sh` from the published pin (a fresh clone of this branch), all
14 build-time check stages passed, and the same image was loaded on ennspark02. It then ran the checks above on the
pair (`raw/release/`, log `rel-smoke.out`):

- Prompt-logprob hash `246cd1331b41`, twice; prefix-cache test 8/8; long tool call intact; loop-repeat repro false.
- Decode prose / code 44.38 / 63.66 tok/s (2.51 / 4.90 tokens per step).
- Cold prefill 2.87 / 10.84 / 20.49 / 40.01 s at 6.4K / 26K / 52K / 105K tokens.
- Images: race correct, 68K conversation turns 2–5 at 0.72–1.00 s, swap and re-send answers identical to the table's
  runs.
- Fill 12/12 in 327.1 s; lowest rank-0 MemAvailable 4.55 GB; 102.46 GB before the KV pool, 146,246 blocks.
- No panics or errors on either rank.

## Receipts

`raw/` holds every receipt: the harness (`ab_up.sh`, `glm_suite.sh`, `glm_final.sh`, `fill_only.sh`,
`prefill_probe.py`, `measure.sh`, `pf_var.sh`), logs (`ab-run2.log` for production, `glm-final.out` for the new
engine) and JSON results. Files named `*prod*` are production, `*gfinal*` the final engine, and `*new*`/`*fnew*`/
`*pfnew*` the intermediate builds before the transpose and sequence-parallel fixes, kept because the text above
cites them. `raw/release/` holds the release image's build log, scripts and smoke receipts; its `rel-smoke.out`
also covers the SparkQwen release smoke that ran after it.
