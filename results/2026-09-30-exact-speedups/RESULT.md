# 2026-09-30: exact prefill and decode kernels, prefix-cache policy, engine fixes

## Change

- **Engine:** [`Enntity/atlas`](https://github.com/Enntity/atlas)
  `sparkglm/atlas-20260930b` @ `dfa4be11`: the previous engine plus 116
  commits from reviewed topic branches.
- **Image:** `ghcr.io/enntity/atlas-sparkglm:227698449651`, built from a fresh
  clone of branch `release/exact-speedups`.
- **Profiles:** both profiles switch on ten engine options. Each is bit-exact
  against the same engine with the option off.

| Option | What it does |
|---|---|
| `ATLAS_GLM_SPARSE_PREFILL_PIPE` | Pipelined sparse-attention prefill kernel |
| `ATLAS_GLM_MOE_PREFILL_PERSIST`, `ATLAS_GLM_MOE_UNPERMUTE_VEC` | Persistent MoE prefill schedule; vectorized unpermute |
| `ATLAS_GLM_INDEX_LOGITS_V2`, `ATLAS_GLM_INDEX_SPLIT` | Faster sparse-index scoring; the index selection split across the two Sparks |
| `ATLAS_GLM_ROUTER_PREFILL_CUTLASS` | Router prefill GEMM through fixed tiles |
| `ATLAS_GLM_MOE_DECODE_M16`, `ATLAS_GLM_MOE_DOWN_ZSKIP` | MoE decode kernels for draft-verify batches |
| `ATLAS_GLM_PC_EVICT` | Snapshot eviction keeps each conversation's newest restore point |
| `ATLAS_GLM_PC_BRANCH` | A restore point where conversations share a prompt prefix |

Credits: `ATLAS_GLM_PC_EVICT` takes its chain-aware eviction from Reederey87's
prefix-cache policy ([glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark),
Apache-2.0, ideas only). `ATLAS_GLM_PC_BRANCH` follows Marconi's branch-point
admission (Pan et al., MLSys 2025, [arXiv:2411.19379](https://arxiv.org/abs/2411.19379)).

Engine fixes that need no option:

- **Sparse-index key race.** A missing barrier let about one index key in
  60,000 prompt tokens be normalized with a wrong mean. Long prompts were not
  reproducible from run to run (shipped separately as image `896f0a60a6b0`).
- **Whole-block prefix sharing.** A request can no longer be handed a partly
  filled cached block that another request still writes.
- **KV pool sizing.** Memory that another process releases while the model
  loads is no longer counted as the engine's own, which oversized the pool.
- **Rank-mirrored KV admission.** A prefill chunk that does not fit is refused
  on both Sparks and retried; before, the pair exited.
- **Kernel race hardening** in the KDA convolution state and three guards.

## Release validation

Two DGX Sparks, 4 × 512K profile, `GPU_MEMORY_UTILIZATION=0.93`, image built
and started by `./start.sh` from a fresh clone. One pass of each test.
Receipts are in `raw/`, the harness in `raw/harness/`, the full log in
`raw/validate5.log`.

| Check | This release | Previous engine, same pair |
|---|---|---|
| Prompt log-probabilities of a 48,875-token prompt, 2 runs | identical hash | same hash (the engine with only the race fix) |
| Cold prompt of 14.6K / 61.0K / 124.8K tokens: time to first token, exact answer | 7.38 / 26.68 / 51.81 s, 3/3 | 7.73 / 29.16 / 62.54 s, 3/3 |
| Four concurrent ~204K-token sessions, 3 turns each | 12/12 exact; first cold 204K prefill 89.3 s (2,290 tok/s) | 12/12; about 110 s (1,860 tok/s) |
| Multi-turn, 45K tokens: time to first token on turns 2–6 | 0.77–1.06 s, 8/8 exact | 0.81 s median |
| Ten-turn conversations: exact answers, warm median | 40/40, 0.79 s | 40/40 |
| Exact-answer checks (concurrent, tool call, partial hit, thinking) | 23/23; long output 41/41 | 23/23 |
| New session sharing a 24K system prompt with earlier ones | 15.7 s | 36.2 s (policy off, see below) |
| Idle session resuming after another session's 17 turns (median) | 2.46 s | 49.6 s (policy off) |
| Frozen matrix, cold (sum of walls) | 157.3 s: C1-16K 14.2, C1-32K 20.3, C2-16K 26.9, C2-32K 42.9, C4-16K 52.9 | 160–161 s |
| Staggered C4 field guide: cold, then three repeats | 56.6 s; 33.0 / 33.9 / 33.0 s | 58.1 s; 32.9 s |
| Single-stream prose decode, 5 × 384 tokens | 32.8–34.5 tok/s | 31.9–32.8 tok/s |
| Panic or ERROR lines, both Sparks | 0 | |
| Lowest host memory available | 1.35 GB on rank 0, 4.99 GB on rank 1 | 1.9 GB on rank 0 at 0.93 |

**8 × 128K profile** (same image, second start): pool 1,440,192 tokens;
exact answers at 14.6K and 61.0K (7.08 and 25.03 s); C4-16K 50.6 s;
1 / 2 / 4 / 8 streams of 512 tokens: 90.4 / 88.1 / 81.2 / 116.1 tok/s
aggregate; no error lines.

## Attribution: one binary, options off and on

Before the release build, the merged engine ran with a development launcher,
one server start per arm, same harness (`raw/rc-round.log`, `raw/rc-round/`).

| Measure | Options off | Eight kernel options on | + the two cache-policy options |
|---|---|---|---|
| Log-probability hash against the previous engine | equal | equal | – |
| Pure prefill of a 65,548-token prompt | 28.81 s | 24.77 s | – |
| Single-stream prose decode (median of 5) | 32.6 tok/s | 34.0 tok/s | – |
| 2 / 4 streams, aggregate | 41.9 / 55.5 tok/s | 46.9 / 58.6 tok/s | – |
| New session sharing a 24K system prompt | 36.2 s | – | 17.6 s |
| Idle session resuming after churn (median) | 49.6 s | – | 2.48 s |
| Normal warm turn (median) | 3.08 s | – | 3.21 s |
| Exact answers in the cache-policy test | 41/41 | – | 41/41 |

## Limitations

- **One pass.** Each number is a single run. The 2- and 4-stream decode rows
  moved by more between passes than the options changed them; read them as
  "not slower".
- **The frozen matrix barely moves.** It is dominated by decoding 400 tokens
  per request at 16–32K context, where this release changes little.
- **Memory headroom.** Rank 0 went down to 1.35 GB available at the end of
  the four-session fill. This start came straight after the image build, so
  other processes held more memory than usual and the pool was 1,677,280
  tokens. A settled start sizes about 1.63M tokens.
- **Greedy decoding is not yet reproducible across request histories.** The
  drafter's first proposal of a request reads a context row left by the
  previous request. Verification is lossless, so the output is valid either
  way, but the same prompt can take a different wording path and speed
  depending on what ran before it. A fix is in test.
- **Not re-measured:** RigMark, the quality probe, and single-stream
  structured and code decode. Those rows in the README still come from the
  first Atlas release.
- **Scrubbed addresses.** Private addresses and host names in the logs and
  the harness were replaced with documentation addresses (192.0.2.x),
  `spark1`/`spark2` and `/home/user`.

- **Edited receipts.** Prompt and filler text that the harness quoted from
  mmastrac's unlicensed repository was replaced by a marker in `raw/harness/`
  (2026-09-30); the numbers were measured with the original text.

## Verify the receipts

```sh
cd results/2026-09-30-exact-speedups && sha256sum -c SHA256SUMS
```
