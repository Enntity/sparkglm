# 2026-09-30: a faster verify step, cheaper warm turns, fairer prefill order

## Change

- **Engine:** [`Enntity/atlas`](https://github.com/Enntity/atlas)
  `sparkglm/atlas-20260930c` @ `9b8160e3`: the previous engine (`dfa4be11`)
  plus the reviewed topic branches below.
- **Image:** `ghcr.io/enntity/atlas-sparkglm:43eed6283b96`, built from a fresh
  clone of branch `release/decode-step`.
- **Profiles:** both profiles switch on ten more engine options.

| Option | What it does | Output |
|---|---|---|
| `ATLAS_GLM_MOE_DECODE_STREAM` | MoE decode kernels that stream expert weights | bit-exact |
| `ATLAS_GLM_DECODE_FUSE` | Fuses small per-layer decode kernels | bit-exact |
| `ATLAS_GLM_DECODE_GEMV_BATCH` | Fewer, earlier-started GEMV launches in the verify step | bit-exact |
| `ATLAS_GLM_DRAFT_TP` | The drafter's MLP and head split across both Sparks | drafts bit-exact |
| `ATLAS_GLM_CMD_RDMA` | Per-step command words over the RDMA pair instead of NCCL | exact |
| `ATLAS_DFLASH_CONF_WIDTH` | Verify width from the drafter's own confidence | lossless; text can differ, as with any width change |
| `ATLAS_GLM_WARM_SKIP_CACHED`, `ATLAS_GLM_WARM_CHUNK_RUN` | A cached prefill chunk skips its zeroing and runs through without waiting for a decode step | exact |
| `ATLAS_PREFILL_SRPT` | A new prefill with fewer tokens left goes ahead of longer ones (never ahead of one waiting 30 s) | scheduling |
| `ATLAS_DFLASH_FIRST_APPEND=none` | The first draft of a request no longer reads a row the previous request left | makes decode independent of request history |

Credits: `ATLAS_DFLASH_CONF_WIDTH` follows knapcio's draft-shape truncation
(`GLM_DRAFT_TRUNC`,
[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
`overlay/glm_draft_trunc.py`) and reuses the survival prefix product of Atlas-Inf's
D-Cut (`crates/spark-server/src/scheduler/mtp_dcut.rs`, arXiv 2607.14647). Its
starting calibration table is knapcio's `overlay/glm_bav_table_seg.json` (MIT).
`ATLAS_GLM_DRAFT_TP` follows MiaAI-Lab's `DFLASH_DRAFT_TP`
([GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks))
and TensorFold's two-rank drafter ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)).

Engine fixes that need no option:

- **Prefill order.** A finished prefill no longer moves the newest one to the
  front of the queue. A 12K request behind a 120K prompt started in 12.6 s
  instead of 55.9 s.
- **Startup agreement.** The two Sparks compare about 60 settings at startup
  and both stop within about 35 s, naming the setting, instead of running with
  a mismatch.
- **`min_tokens`** now bans end tokens for prompts that prefill in one chunk.
- **Head-only cache insert.** At the end of a request only the head Spark cached
  the reply's blocks; now neither does unless both can.

## Release validation

Two DGX Sparks, 4 × 512K profile, `GPU_MEMORY_UTILIZATION=0.93`, image built
and started by `./start.sh` from a fresh clone. One pass of each test.
Receipts are in `raw/`, the harness in `raw/harness/`, the full log in
`raw/validate6.log`.

| Check | This release | Previous release, same pair |
|---|---|---|
| Prompt log-probabilities of a 48,875-token prompt, 2 runs | identical hash `8c75d2886794`, equal to the previous release's | same hash |
| Cold prompt of 14.6K / 61.0K / 124.8K tokens: time to first token, exact answer | 7.06 / 25.02 / 49.13 s, 3/3 | 7.38 / 26.68 / 51.81 s |
| Four concurrent ~204K-token sessions, 3 turns each | 12/12 exact; first cold 204K prefill 88.6 s | 12/12; 89.3 s |
| Multi-turn, 45K tokens: time to first token on turns 2–6 | 0.67–0.93 s, 8/8 exact | 0.77–1.06 s |
| Ten-turn conversations: exact answers, warm median | 40/40, 0.67 s | 40/40, 0.79 s |
| Exact-answer checks (concurrent, tool call, partial hit, thinking) | 23/23; long output 41/41 | 23/23; 41/41 |
| Cache-policy test: exact answers | 41/41 | 41/41 |
| Single-stream prose decode, 5 × 384 tokens | **40.2–44.2 tok/s** | 32.8–34.5 tok/s |
| 2 / 4 streams, prose, aggregate | 53.3 / 67.5 tok/s | 46.9 / 58.6 tok/s (earlier run) |
| Frozen matrix, cold (sum of walls) | **146.0 s**: C1-16K 15.0, C1-32K 20.5, C2-16K 27.3, C2-32K 38.5, C4-16K 44.7 | 157.3 s |
| Staggered C4 field guide: cold, then three repeats | 52.7 s; 27.3 / 27.6 / 27.8 s | 56.6 s; 33.0 / 33.9 / 33.0 s |
| Panic or ERROR lines, both Sparks | 0 | 0 |
| Lowest host memory available | 1.77 GB on rank 0, 2.55 GB on rank 1 | 1.35 GB on rank 0 |

**8 × 128K profile** (same image, second start): pool 1,436,352 tokens;
exact answers at 14.6K and 61.0K (6.82 and 24.67 s); C4-16K 49.9 s;
1 / 2 / 4 / 8 streams of 512 tokens: 95.8 / 91.1 / 85.8 / 130.5 tok/s
aggregate (previous release 90.4 / 88.1 / 81.2 / 116.1); no error lines.

## Attribution: one binary, options off and on

The same merged engine, one server start per arm, with
`ATLAS_DFLASH_FIRST_APPEND=none` in every arm so greedy text is comparable
(`raw/arms/go_decode.log`):

| Arm | Single-stream prose (median) | 2 / 4 streams | Greedy text against the first arm |
|---|---|---|---|
| New decode options off | 33.8 tok/s | 45.1 / 57.7 | – |
| The five exact decode options | 36.3 tok/s | 50.2 / 62.0 | identical, same tokens per step |
| … and verify width from confidence | 39.8 tok/s | 53.2 / 67.5 | differs |

Warm-turn options on the release-candidate engine (`raw/arms/go_rc4.log`),
engine time to first token, identical logits on all 48 warm turns:

| Context | Off | On |
|---|---|---|
| 16K | 593 ms | 556 ms |
| 45K | 694 ms | 575 ms |
| 45K while another stream decodes | 1,206 ms | 586 ms |

## Limitations

- **One pass.** Each number is a single run; 2- and 4-stream aggregates move
  by about 3% between starts of the same binary.
- **Width changes text.** With the confidence rule on, greedy text can differ
  from the previous release on the same prompt, as it already could when the
  adaptive width changed. Verification stays lossless.
- **Tokens per step are lower.** On a 39-prompt natural-text set they fell 2.6%,
  as expected when fewer rows are verified; tokens per second rose.
- **Not re-measured:** RigMark and the quality probe. Those README rows still
  come from the first Atlas release.
- **Scrubbed addresses.** Private addresses and host names in the logs and the
  harness were replaced with documentation addresses (192.0.2.x),
  `spark1`/`spark2`/`spark3` and `/home/user`.

- **Edited receipts.** Prompt and filler text that the harness quoted from
  mmastrac's unlicensed repository was replaced by a marker in `raw/harness/`
  (2026-09-30); the numbers were measured with the original text.

## Verify the receipts

```sh
cd results/2026-09-30-decode-step && sha256sum -c SHA256SUMS
```
