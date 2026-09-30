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
- With caching on, the engine disables its fused prefill/decode path. Cold
  C4-16K is about 6% slower than in the release without caching.
- The first start after installing or updating compiles CUDA kernels. On our
  pair this briefly used about 2 GB more host memory than steady state.

## Performance gaps

We compared against the published RiNGSiDE vLLM TP2 results, not a run on our
pair. Atlas is behind on:

- single-stream decode: about 7% on code and 20% on prose;
- low-concurrency short code: 36.9 against 44.0 tok/s with one stream, 76.7
  against 97.3 with six;
- long cold prefill: 28.3 against 25.7 s at 64K.

Atlas is ahead on newcomer time-to-first-token under load and on 8K prefill.

## Behaviour

- Requests with thinking disabled still get a short, low-effort reasoning
  block, returned in `reasoning_content`. The reasoning budget is enforced, so
  structured answers are not trapped inside reasoning.
- The quality probe scored 11/12 on two-hop needles in the recorded run; the
  same engine scored 10–12/12 across earlier runs. Arithmetic scored 40/40.
- Sparse attention uses GLM-5.3's full 2051-candidate selection. The vLLM
  version's 2048-candidate approximation does not apply here.

## Operations

- The API has no authentication and listens only on rank 0's loopback.
- GB10 memory is shared with the host. Running other large workloads on
  either Spark while serving can hang the host.
- Both shipped profiles need the DFlash2 drafter, which is licensed for
  non-commercial use only. No profile without it is shipped yet.
- The image is built for arm64 and SM121 (GB10) only.
