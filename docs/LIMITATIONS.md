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
- Under real mixed traffic at 0.93, rank 0 dropped below our 1 GiB memory guard
  about hourly (2026-10-02 to 10-04), so our production setting is now 0.91.
  At 0.91 the pool is about 1.14M tokens, or about 1.46M with
  `DISPLAY_CARVEOUT=1`, which adds KV without using system memory.
- `DISPLAY_CARVEOUT=1` needs `CAP_SYS_ADMIN` on the containers (dropped before
  the server starts), a validated NVIDIA driver (580.173.02 or 580.178.04) and
  a headless Spark. It has been measured on our pair only.
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
([decode and prefill](../results/2026-10-04-rigmark/RESULT.md) on this release,
[concurrency and staggered arrivals](../results/2026-09-30-rigmark/RESULT.md)
on 2026-09-30), Atlas is ahead of RiNGSiDE's TP2 row on single-stream decode,
cold prefill at 8K-64K and newcomer time to first token under staggered
arrivals. Still behind or unmeasured:

- mmastrac's TP2 row (63.1 / 36.6 / 89.1 tok/s code / prose / structured,
  published 2026-09-30) against our 64.5 / 35.3 / 88.4: about 2% ahead on
  code, about 4% behind on prose and 1% behind on structured;
- cold prefill at 64K took 25.13 s on this release against 24.38 s on
  2026-09-30 (3.1% slower, both runs); 8K and 32K are unchanged. The cause is
  not yet known;
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

## Structured output

- `response_format` (`json_schema` or `json_object`) is enforced token by token
  with a grammar, including during speculative decoding. A strict schema the
  engine cannot enforce is refused with HTTP 400 (an error event when
  streaming). It never silently falls back to free text.
- With speculative decoding (`ATLAS_GLM_STRICT_SPEC=1`, on in both profiles),
  the tokens chosen at verify positions are greedy, as for every other
  speculative request on this engine: temperature, top_p and repetition
  penalties apply only to the tokens decoded serially.
- A very long strict list at temperature 0 can fall into repetition. Asked for
  60 items, the model produced 121 and degraded into nonsense until
  `max_tokens`. The JSON stays valid, but the content is not usable. Requests
  at our clients' settings (temperature 0.1-0.25, bounded or keyed objects)
  were not affected.
- Every `json_schema` is enforced as strict, including `"strict": false`.
