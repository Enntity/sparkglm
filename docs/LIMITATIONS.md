# Limitations

What to know before relying on SparkGLM.

## Scope of the measurements

- We tested one pair of DGX Sparks on one cable. Other firmware, drivers,
  cables or cooling may give different numbers.
- The vLLM and Mia comparison runs were recorded earlier on the same pair.
  They were not alternated with the Atlas runs.
- Each README row comes from the release its receipt names. RigMark decode,
  cold prefill, staggered arrivals, the staggered C4 field guide, the Matrix
  and the four-session 204K fill were re-measured on the current release,
  2026-10-09 RC2. RigMark short-code concurrency was also re-measured on RC2
  (47.7 tok/s at one stream, 69.5 at four); the earlier 2026-09-30 receipt is
  kept for history. The quality probe's first recorded run is from the first
  Atlas release, but the 500-item arithmetic (491/500) and two-hop 24K needle
  (24/24) sets were re-run on RC2.
- A cold image build has not been timed. Our builds reused BuildKit caches.

## Not yet measured

- **Long context:** on 2026-10-09 RC2 we ran four concurrent ~204K-token
  sessions with cached follow-ups (12/12 and 8/8 exact) and a two-hop 24K
  needle probe (24/24), but we have not exercised a single request anywhere
  near the profile's full 1,048,576-token window under load. On an earlier
  engine commit, before the thinking-budget fix, four concurrent 190K requests
  and one 500K request completed with at least 8.3 GB of host memory free.
- **Chunked prefill and endurance:** prefill is not row-invariant. A prompt
  long enough to prefill in several chunks (over 8K tokens) can still differ
  when the engine is busy, and long-context endurance, cancellation under load
  and multimodal work at full context are not measured.

## Capacity and prefix caching

- All requests share one KV pool, which also holds the prefix cache. Unsplit,
  it is about 600K tokens at the default 0.88 memory utilization and about
  1.13M at 0.91 on dedicated Sparks. With the KV shard (`./start.sh`'s default)
  it holds about 1.8x that: 2.61M tokens at 0.91 with the display carveout, or
  2.36M with the disk prefix cache. A request needs pool room for its whole
  context, so four full 1M requests do not fit at once; later requests wait
  for room.
- At 0.93 on dedicated Sparks the pool is about 1.63M tokens, and rank 0 went
  down to 1.35 GB of available memory at the end of a four-session 204K fill.
- At 0.92, a Spark sharing the host with other services went down to 3.1 GB of
  free memory. GB10 hosts can hang when memory runs out, so raise the setting
  only on Sparks that run nothing else.
- Under real mixed traffic at 0.93, rank 0 dropped below our 1 GiB memory guard
  about hourly (2026-10-02 to 10-04), so our production setting is 0.91. At
  0.91 the pool is about 1.13M tokens unsplit, or about 1.38M with
  `DISPLAY_CARVEOUT=1`, which adds KV without using system memory. On the
  2026-10-09 RC2 release, with the KV shard, the carveout and the disk prefix
  cache, the pool is about 2.36M tokens and a four-session 204K fill left
  rank 0 at 4.8 GB free at worst. The GPU
  does not cache the carveout in L2, so only the latent KV pools go there;
  a cacheable mapping that could also take the index buffers is tracked in
  [#32](https://github.com/Enntity/sparkglm/issues/32).
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

These come from two different kinds of comparison and should not be read as
one table.

**Same-pair.** Mia's TensorFold v1.10 was run on our pair for the current release
([receipts](../results/2026-10-09-rc2/RESULT.md)). On that probe Atlas is
ahead on staggered C4 (~16K: 49.9 s vs 60.2; ~32K: 77.3 s vs 100.1) and on
the Matrix (136.3 s vs 166.9), while TensorFold is faster at single-stream
decode on the same probe prompts (about 49.8 vs 42.2 tok/s prose, 68.5 vs 54.6
code). TensorFold is exl3-based, whose experts are about 8% smaller per rank
than our NVFP4 experts, and our MoE decode kernels already read weights at the
GB10 bandwidth roofline.

**External, published not re-run.** Against the published RiNGSiDE vLLM TP2
figures on their RigMark cell set, Atlas is ahead on single-stream decode,
cold prefill at 8K-64K and newcomer time to first token under staggered
arrivals. Their short-code result is 97.3 tok/s at six streams; RC2 measured
69.5 tok/s at four streams. The default 4 x 1M profile serves four requests
at a time, so these are different concurrency settings.
The 8 x 128K profile, which serves eight, has not been measured with RigMark.
Older comparisons, including the mmastrac vLLM TP2 figures, remain in their
dated result bundles.

## Behaviour

- Requests with thinking disabled still get a short, low-effort reasoning
  block, returned in `reasoning_content`. The reasoning budget is enforced, so
  structured answers are not trapped inside reasoning.
- RC2 scored 491/500 on the larger arithmetic set and 24/24 on two-hop 24K
  needles. Arithmetic scores across the tested engine configurations ranged
  from 484 to 495; single runs do not establish a quality ranking. Earlier
  small-probe scores remain in their dated result bundles.
- Four greedy prompts run one at a time at C1, then all four together
  at C4, twice, gave the same output every time: each C4 run matched the C1
  output and the two C4 runs matched each other (batched verify, adaptive
  draft width, copy drafts and the prefix cache on;
  `raw/rc2-spark-rc2-f2b805e7.json`, against 0 of 4 on the 2026-10-09 morning
  release). This is the measured four-prompt result, not a general proof.
  Prefill is not row-invariant, so a prompt long enough to prefill in several
  chunks (over 8K tokens) can still differ when the engine is busy (see
  "Not yet measured").
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
