# 2026-10-14: image prompts use the prefix cache (engine sparkglm/atlas-20261014-vpc)

The image is `03f0ecc85c27`. It runs engine `8180ae56` on `sparkglm/atlas-20261014-vpc`: the upstream candidate
`upstream/glm53-flash-20261014-vpc` at `28f64b91` (RC4.2's `d8a55e0f` plus three vision commits), the unchanged
SparkGLM-only layer and the refusal-content fix. Text prompts run exactly as on RC4.2. Not yet published to GHCR or
deployed.

## What changed

- **Image prompts hit the prefix cache.** Every image is the same run of placeholder tokens, so image prompts used to
  bypass the cache: once an image entered a long conversation, every turn re-prefilled the whole history and
  re-encoded every image (92–97K tokens, about 35 s to first token per turn). The cache key now replaces each
  placeholder with an id derived from its image's SHA-256 content hash. A different image never shares KV, the same
  image hits, and text before the first image keys as before (Atlas issue 58).
- **Only images a prefill computes go through the vision encoder,** chunk by chunk, with the vision state held per
  request, so concurrent image prompts cannot read each other's encoder rows. On RC4.2 a long image prompt still
  prefilling could take a second request's image; the reproduction answers with the wrong image there and correctly
  here.
- **The second Spark loads no vision tower.** Rank 0 encodes and sends the rows; rank 1's footprint before the KV pool
  drops by 2.2 GB.
- A model reply in an image conversation is not cached (decode and a re-prefill write different MRoPE positions), so a
  next turn reuses the previous prompt and re-prefills only the reply and the new message.

## Release checks

Two DGX Sparks (ennspark03/04), production settings from the shipped `4x1m` profile: 0.91, KV shard, NVMe prefix and
SSM tiers, 1M context. Both images ran back to back on the same pair, the new one against RC4.2's production image
`ed7548dfe442` (`5ec57d55`). On this pair the display carveout is not available (03 runs a desktop session) and other
resident services left rank 0 less free memory than production's ennspark01, so every pool below is smaller than
production's and set by rank 0's free memory; the comparisons are like for like.

**Image build.** All 13 build-time check stages passed, including a new vision-cache stage (content keys, the
receive-only rank-1 tower, the per-chunk encode and its stream ordering, the rank wire).

**Text output unchanged.**
- Prompt-logprob hash of the fixed 48,875-token text: `246cd1331b41` on both images, RC4.2's published hash
  (`raw/release-checks/lp-rel-*.json`).
- The 500 arithmetic problems and 24 two-hop needles: 524 of 524 outputs byte-identical to the production image on the
  same pair and to RC4 on ennspark01/02; scores 491/500 and 24/24 (`raw/bench-*/qbig-*.jsonl`).
- Quality probe: 39/40 arithmetic, 12/12 needles on both.

**Images.**
- A 68K-token codebase conversation with one image: turn 1 cold 27.8 s; turns 2–5 start in 0.72–1.02 s with 68,128–68,240
  tokens cached and the image not re-encoded (`raw/release-checks/conv-rel.json`). On the production image every turn
  re-prefills (27 s).
- Swapped image order, a different image at the same position, a re-sent image: all answered correctly; only
  identical content is reused (`swap-rel.json`).
- Interleaved image prompts (A: blue square at the end of 47,772 tokens; B: green triangle sent 3 s later): A answers
  "Blue square", B "green triangle", B not held (TTFT 8.8 s) (`race-rel.json`).
- NVMe tier: after five rounds of 204K fill traffic evicted it, the image conversation's prefix (4,266 blocks, 68,256
  tokens) came back from disk in 102 ms with 0 failures and the same answer (`nvme-rel.json`, `go_rel.log`).

**Serving.**
- Long tool call: one `write_file` of 3,516 tokens returns intact.
- Prefix-cache test (45K multi-turn and replay): 8/8 (`pc-rel-new.json`).
- 20-minute soak, four concurrent workers (two image conversations behind 2K–30K tokens of code, 2,400-record
  retrieval, short arithmetic): 449 requests, no errors. Image answers 251/251; retrieval 48/49 (one wrong record);
  arithmetic 145/149, the 4 being replies whose short reasoning used up the 12-token cap before the number, which the
  production image does too (`soak-rel.json`, `arith-*.json`).
- Fill check, four concurrent ~204K sessions × 3 turns, five rounds of fresh documents: 57/60 on the new image and
  57/60 on the production image, the same three requests with the same wrong answers (`fill-rel-fill*.json`,
  `fill-fab-prod-fill*.json`). All three are cached later turns; sent cold, both images answer them correctly
  (`replay-*.json`). This is an existing warm-path behaviour of the production engine, not a regression; it is
  being investigated separately.
- Lowest MemAvailable (1 s samples): rank 0 2.87 GiB during the fills; rank 1 8.53 GiB. memguard (1 GiB) never fired.
- Rank 1 before the KV pool: 100.3 GB against 102.5 GB on the production image (`go_rel.log`).

## Benchmarks (ennspark03/04, same settings, back to back)

The RC4 release suite (`raw/summarize_bench.py` reproduces RC4's published table from its receipts).

| | This release | Production (RC4.2) |
|---|---:|---:|
| C1 decode, 12 prompts × 256 forced tokens, prose / code, tok/s (`raw/release-checks/multi-rel-*.json`) | 44.06 / 63.42 | 43.68 / 63.84 |
| Strict prose bursts, 8 prompts × 384 tokens, tok/s (tok/step) | 46.9 (2.52) | 46.8 (2.52) |
| RigMark decode, code / prose / structured, tok/s (median of 3) | 66.7 / 37.2 / 90.4 | 68.2 / 38.0 / 89.7 |
| RigMark cold prefill 8K / 32K / 64K, TTFT (s) | 3.63 / 13.35 / 25.51 | 3.58 / 13.44 / 25.94 |
| RigMark staggered arrivals, prefill first: newcomer TTFT at 2 / 4 (s) | 3.05 / 4.38 | 3.58 / 3.81 |
| RigMark short code, 1 / 2 / 3 / 4 streams aggregate, tok/s | 48.2 / 58.3 / 63.3 / 68.7 | 46.8 / 56.5 / 60.4 / 68.8 |
| Staggered C4 field guide, ~16K input (median of 5), s | 51.7 | 51.8 |
| Staggered C4 field guide, ~32K input (median of 3), s | 79.8 | 80.9 |
| Matrix (C1/C2 16K and 32K, C4 16K; sum of walls), s | 140.4 | 143.6 |

- Speed is unchanged: the equal-text rows (C1 decode, bursts, tokens per step) agree within 1%. RigMark samples, so its
  text and lengths differ between runs; its rows move both ways within run-to-run spread.
- The numbers sit below RC4's on ennspark01/02 (for example RigMark code 72.3, field guide 16K 49.5 s) for both images:
  that is the pair (no display carveout, smaller pool, other residents on 03), not the release.

## Not covered

- Video and partial multi-image reuse ran in unit tests only. An image straddling a chunk boundary is re-encoded per
  chunk: cheap for a still, several encoder passes for a long video.
- The pool on the production pair: rank 1's freed 2.2 GB should make rank 0 the limit there (about +3% KV); read it from
  the boot log at deployment.

## Verify the receipts

```sh
cd results/2026-10-14-vision-prefix-cache && sha256sum -c SHA256SUMS
```
