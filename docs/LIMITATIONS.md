# Limitations

What to know before relying on SparkGLM.

## Scope of the measurements

- We tested one pair of DGX Sparks on one cable. Other firmware, drivers,
  cables or cooling may give different numbers.
- The vLLM and Mia comparison runs were recorded earlier on the same pair.
  They were not alternated with the Atlas runs.
- The decode, prefill, RigMark and quality numbers come from the first Atlas
  release (no prefix caching, same kernels). The prefix-caching release
  re-measured the matrix, the staggered field guide and multi-turn behaviour.
- A cold image build has not been timed. Our builds reused BuildKit caches.

## Not yet measured with the recipe image

- **Long context:** we have not measured it with the recipe image. On the same
  engine commit, before the thinking-budget fix, four concurrent 190K requests
  and one 500K request completed with at least 8.3 GB of host memory free.
- **Endurance, cancellation under load, and multimodal work at full context:**
  not measured.

## Capacity and prefix caching

- All requests share one KV pool, which also holds the prefix cache: about
  600K tokens at the default 0.88 memory utilization, and 1.39M at 0.92 on
  dedicated Sparks. A request needs pool room for its whole context, so four
  full 512K requests do not fit at once; later requests wait for room.
- At 0.93 on dedicated Sparks the pool is about 1.63M tokens, and rank 0 went
  down to 1.35 GB of available memory at the end of a four-session 204K fill.
- At 0.92, a Spark sharing the host with other services went down to 3.1 GB of
  free memory. GB10 hosts can hang when memory runs out, so raise the setting
  only on Sparks that run nothing else.
- Prefix caching keeps 16 recurrent-state snapshots, one per recent request.
  A conversation idle for longer than that is re-prefilled from its deepest
  remaining cached block, or from scratch.
- Only block-aligned snapshots written during prefill are restored
  (`ATLAS_MARCONI_PREFILL_ONLY`). The engine's other snapshot kinds gave a
  wrong retrieval answer on a cache hit; see the prefix-caching result.
- Cache matches stop at the last whole 16-token block
  (`ATLAS_PREFIX_SUBBLOCK=0`). Before 2026-09-30 the profiles also let a
  request share a cached block that was only partly filled. Two requests could
  then write the rest of the same block, for example a retry of an identical
  prompt or the second of `n > 1` choices, whenever the prompt length was not
  a multiple of 16. GLM restores from block-aligned snapshots anyway, so
  turning this off costs nothing.
- The prefix cache on disk (`PREFIX_CACHE_DIR`, off by default) has been
  measured in one pass, on 25K-token conversations. Long contexts, many
  restores at once, the cost to the request whose prefill pushes blocks out,
  and endurance are not measured.
- The engine refuses to start when the directory is on tmpfs, ramfs or
  overlayfs, cannot be written, or cannot hold the KV half of the size, and
  when the two Sparks' settings differ. Its host memory comes out of the KV
  pool (see the README).
- With caching on, the engine disables its fused prefill/decode path. Cold
  C4-16K is about 6% slower than in the release without caching.
- The first start after installing or updating compiles CUDA kernels. On our
  pair this briefly used about 2 GB more host memory than steady state.

## Performance gaps

We compare against the published RiNGSiDE and mmastrac vLLM TP2 results, not
runs on our pair. On RigMark with RiNGSiDE's cell set
([results/2026-09-30-rigmark](../results/2026-09-30-rigmark/RESULT.md)), this
release is ahead of RiNGSiDE's TP2 row on single-stream decode, cold prefill
at 8K-64K and newcomer time to first token under staggered arrivals. Still
behind or unmeasured:

- mmastrac's TP2 row (63.1 / 36.6 / 89.1 tok/s code / prose / structured,
  published 2026-09-30) against our 62.9 / 34.0 / 87.5: level on code, about
  7% behind on prose;
- short code at six streams: 97.3 tok/s for RiNGSiDE against our 62.0, because
  the 4 x 512K profile serves four sequences at a time; the 8 x 128K profile,
  which serves eight, has not been measured with RigMark.

## Behaviour

- Requests with thinking disabled still get a short, low-effort reasoning
  block, returned in `reasoning_content`. The reasoning budget is enforced, so
  structured answers are not trapped inside reasoning.
- The quality probe scored 11/12 on two-hop needles in the recorded run; the
  same engine scored 10–12/12 across earlier runs. Arithmetic scored 40/40.
- Greedy decoding is not yet reproducible across request histories. The
  drafter's first proposal of a request reads a context row left by the
  previous request, so the same short prompt can take a different wording
  path, at a different speed, depending on what ran before it. Every token
  is still verified by the full model. A fix is in test.
- Long prompts were not reproducible before 2026-09-30. A missing barrier in
  the kernel that normalizes sparse-index keys let one in roughly 60,000
  prompt tokens store a key normalized with a wrong mean. The key stayed in
  the index cache, so the same long prompt could select slightly different
  tokens from run to run and on the two Sparks. Answers in our retrieval
  tests stayed correct. With the fix, four runs of a 50K-token prompt hash
  identically at every traced stage on both ranks.
- Sparse attention uses GLM-5.3's full 2051-candidate selection. The vLLM
  version's 2048-candidate approximation does not apply here.

## Operations

- The API has no authentication and listens only on rank 0's loopback.
- GB10 memory is shared with the host. Running other large workloads on
  either Spark while serving can hang the host.
- Both shipped profiles need the DFlash2 drafter, which is licensed for
  non-commercial use only. No profile without it is shipped yet.
- The image is built for arm64 and SM121 (GB10) only.
