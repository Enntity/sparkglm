# Improving cold prefill tok/s — analysis and plan

**Date:** 2026-08-29 · **Scope:** cold-prefill throughput on the GLM-5.3-Flash EXL3 2× DGX Spark serve.
**Baseline receipts:** `docs/cold-prefill.md` (2026-08-29 ladder). Decode report: `docs/glm53-flash-report.md`.
Companion to `docs/GRAPH-AND-NOPE-PLAN.md` (decode) — this is the prefill twin.

---

## 1. Where we are (measured)

Cold prefill ladder, live 1M serve, temp 0, unique salt (APC can't cheat), one request at a time:

| Rung | TTFT | Prefill tok/s |
|---|---:|---:|
| ~8k cold | 10.36 s | **772** |
| ~12k cold | 13.38 s | **896** |
| ~16k cold | 17.91 s | **893** |
| ~100k cold | 105.6 s | **947** |
| ~256k cold | 273.4 s | **936** |
| ~300k cold | 323.2 s | **928** |
| ~8k follow-up (APC hit 7168) | 1.30 s | 6167 on 836 compute tokens |

Two structural facts fall out of the ladder:

1. **Steady-state cost ≈ 1.05–1.10 s per engine step.** 100k tokens at MNBT=1024 is ~98
   chunks → 105.6 s ≈ **1.08 s per 1024-token chunk**. The same per-chunk cost explains 256k
   (250 chunks) and 300k (293 chunks). Throughput is nearly flat from 100k→300k, so the
   per-chunk cost is dominated by terms **linear in new tokens** with only a small
   length-dependent term (the indexer does *not* blow up with KV length on these rungs).
2. **Short rungs are worse** (772 at 8k): fixed per-request overhead (tokenize, template,
   APC hash, queue, first-chunk ramp) amortizes poorly below ~12k tokens. That is the
   TTFT a new chat session actually feels.

APC follow-ups are already fast (6.2k tok/s on the compute tail) — the problem is purely
**cold compute**.

## 2. Roofline: what a prefill chunk *should* cost

From `$MODEL_DIR/config.json` (`glm5_next`): 45 layers = **34 KDA linear-attention + 11
NoPE sparse-MLA (GLM_NSA)**; MoE from layer 3: **288 routed experts, top-8**, one shared;
`hidden 4096`, `moe_intermediate 2048`; vocab 154,880. Active params ≈ **19.5 B/token**:

| Component | Params/tok | GFLOP/tok | Share |
|---|---:|---:|---:|
| Routed experts (42 × 8 × 25.2M) | 8.46 B | **16.9** | 47% |
| KDA layers (34 × ~170M) | 5.8 B | **11.6** | 33% |
| Sparse MLA (11 × 117.5M) + indexer | 1.66 B | 3.3 | 9% |
| Shared experts (42 × 25.2M) | 1.06 B | 2.1 | 6% |
| Dense MLP ×3 | 0.45 B | 0.9 | 2.5% |
| lm_head (154,880 × 4096) | — | 1.27 | 3.5% |
| **Total** | **~19.5 B** | **~36** | |

At **947 tok/s** that is **34 TFLOPS combined ≈ 17 TFLOPS per Spark ≈ 14% of the GB10
BF16 dense peak (~125 TFLOPS)**. There is headroom; the question is which wall we hit first.

**Per-chunk anatomy at MNBT=1024 (the 1.08 s), ranked:**

| Wall | Estimate | Why |
|---|---:|---|
| Routed-expert weight streaming | **≥ 330 ms** | top-8 × 1024 tokens = 8,192 expert-slots over 288 experts → **every expert activates** (~28 rows each, hot ones >128). Bytes touched/GPU = 42 layers × 288 × 6.3 MB (TP half) ≈ **75.6 GB** → 330+ ms at ~230 GB/s effective LPDDR5x. Arithmetic intensity ≈ 112 FLOP/B < the 458 FLOP/B bf16 ridge → **memory-bound at this chunk size**. |
| KDA chunked scan (34 layers) | **~250–600 ms** | 6.1 TFLOP/GPU per chunk; Triton GDN kernels on sm121a are not deeply tuned; exact rate unknown until profiled (P0). |
| Sparse MLA + GLM_NSA indexer | unknown | `FLASHINFER_MLA_SPARSE_SM120`, 576-wide padded, topk 2048, `fp8_ds_mla`. Known to have a "large per-step cost" (`overlay/patch_scheduler_decode_floor.py`); smem limits force MNBT ≤ ~4096 (**8192 is unsafe — indexer smem**, lab notes §5). |
| TP allreduce over CX7 | ~15–25 ms | one AR/layer × 45 × (1024×4096×2 B) ≈ 378 MB per step over a **200 Gb/s** port (ethtool-confirmed) ≈ 15–20 ms bus + latency. Real but minor. |
| Host syncs + launch overhead | up to ~50 ms | `apply_exl3_fused_moe` falls back to `apply_exl3_python_loop` for **any expert with >128 rows** → `.tolist()` **host sync per layer** plus per-expert GEMM launches (`overlay/exl3.py`, `TEMP_ROWS_FUSED = 128`). With skewed routing this fires often at 1024-token chunks. |

So: at today's chunk size the MoE is a **bandwidth problem** (all 288 experts read per
chunk regardless of batch size), KDA is the big unknown, and the fat-expert fallback
burns both a host sync and kernel efficiency.

## 3. Plan (ranked by expected gain × effort)

### P0 — Profile before touching anything (0.5 day, read-only, decides everything)

One torch-profiler (or nsys) capture of a 16k and a 100k cold prefill, attributing the
1.08 s/chunk to: `exl3_moe` fused vs `LinearEXL3` loop, KDA/GDN Triton kernels,
`flashinfer` sparse MLA + indexer, NCCL AR, elementwise/glue. Plus a one-line counter in
`apply_exl3_fused_moe` logging **how often the fat-expert fallback fires and the max
per-expert row count** at MNBT=1024 (expect: hot experts >128 rows routinely).

Do not skip this: P3/P4 ranking flips depending on whether KDA or indexer dominates.

### P1 — Chunk-size ladder A/B (half a day, no code)

The single cheapest lever. Today's 1024-token chunk is *far* below the compute-bound
regime: per-expert M≈28 keeps the MoE memory-bound and every fixed per-step cost is paid
~98× for a 100k prompt.

- Ladder `MAX_NUM_BATCHED_TOKENS`: **1024 → 2048 → 3584 → 4096** (stop; **8192 known-unsafe:
  indexer smem**, lab notes §5).
- Prefer **3584** as the sweet-spot candidate: it equals the hybrid APC/KpoolTail page
  (4×896), so chunk boundaries align with block-aligned prefix-cache pages — chunked
  writes land exactly on hybrid page boundaries instead of straddling them. May also
  *improve* tail-hit behavior; re-run the APC follow-up rows of `docs/_run_cold_prefill.py`
  after each rung.
- Keep `GLM53_MIXED_PREFILL_CHUNK=skip`. Bigger chunks make a mixed step worse for
  concurrent decoders (a 3584-token sparse-MLA chunk stalls decode for seconds, not 1.5 s)
  — the skip policy becomes *more* important, not less. Document the trade: cold-prefill
  tok/s up, concurrent-decode interleave coarser.
- Guardrails per rung: `/health` 200, content `OK`, no NaN, APC isolation rows intact.

Expected: **+10–30%** cold prefill (fixed-cost amortization + fewer fat-expert firings).
Measured via the same cold ladder; also record 8k rung (fixed-overhead share shrinks).

### P2 — Make the fused MoE path chunk-size-robust (1–2 days; prerequisite for P1 to shine)

`apply_exl3_fused_moe` (`overlay/exl3.py`) caps the kernel temp at `TEMP_ROWS_FUSED=128`
rows/expert and falls back to a **host-synced python loop** for fat experts. Three options,
in increasing ambition:

1. **Bump `TEMP_ROWS_FUSED`** to 512–1024. Temps are `(concurrency, rows, dim)` fp16 ×4
   buffers — at rows=1024, concurrency 8 that is ~400 MB, paid out of the KV pool at util
   0.87 (watch the logged pool shrink; `CG_ESTIMATE=0` interplay). Cheapest fix; keeps
   `.tolist()` for the rare still-fatter expert.
2. **Row-tiling on GPU (recommended):** chunk the *sorted* expert-token buffer into
   128-row tiles and launch `exl3_moe` per tile with correct `expert_count` offsets.
   Bounded temps, zero host syncs, works at any chunk size. All GPU: argsort +
   `scatter_add_` already are.
3. **Dequant-once grouped GEMM for large T** (compute-bound regime, M≥256/expert):
   dequant trellis→bf16/fp8 once per activated expert, then a CUTLASS/`torch._grouped_mm`
   grouped GEMM. Dequant traffic amortizes only when M is large, so gate it on
   `tokens/chunk > ~4k`. Biggest ceiling at 3584–4096 chunks, most work.

Expected: **+5–20%** at today's chunk (removes host syncs + loop-GEMMs), and it unlocks
P1's larger rungs. Keep `num_active=-1` graph-safety from the decode work.

### P3 — KDA/GDN scan uplift (1–3 days; the big unknown until P0)

KDA is 33% of FLOPs and pure Triton on sm121a. If P0 shows GDN kernels at <20% MFU:

- Autotune chunk sizes (`TLA` scan chunk 64 vs 128) and head-dim tiling for sm121a;
  ensure the persistent Triton cache (already shipped) covers prefill shapes, not just
  DFlash2 decode shapes — warm them post-`/health` like `scripts/boot-shape-warmup.sh`
  does for drafter shapes.
- Check state dtype path (`mamba_cache_dtype=auto`) and whether the scan accumulates in
  fp32 SIMT where `tl.dot` tensor-core paths exist.

Expected: **0–30%** (pure function of what P0 finds).

### P4 — Short-context attention fast path + selective autotune (1 day)

The 8k rung (772 tok/s, 10.4 s TTFT) is the one users feel on every new session.

- For context ≤ `index_topk` (2048), GLM_NSA sparse attention degenerates to dense.
  Check whether the FlashInfer SM120 kernel auto-takes a dense path; if not, a dense-MLA
  fallback under a threshold (e.g. ≤8k) could lift the short rungs specifically.
- Revisit `--no-enable-flashinfer-autotune`: it is off because a full autotune OOMs this
  UMA. Try tuning **prefill shapes only** (boot with small `MAX_MODEL_LEN`, capture the
  autotune cache, persist it, then serve 1M with the cache preloaded). Decode report
  priced this at 0–10%; for prefill the sparse kernel has never been tuned.

Expected: **+0–15% on ≤16k rungs**, ~0 on ≥100k.

### P5 — Comm micro-wins (0.5 day + a cable; small for prefill, more for decode)

- NCCL: `NCCL_PROTO=LL128`, `NCCL_ALGO`, `NCCL_MIN_NCHANNELS` for the 8.4 MB/layer
  messages over RoCE; measure with `NCCL_DEBUG=INFO` timings.
- **Second rail:** head's `enp1s0f0np0` and worker's `enp1s0f1np1` are unused (this kit
  pins head-f1 ↔ worker-f0). A second QSFP cable + `NCCL_IB_HCA` list + per-rail GIDs
  doubles inter-Spark bandwidth. Worth ~2–5% prefill; materially helps decode AR too.

### P6 — fp8 for the non-routed GEMMs (3–5 days, policy-gated)

GB10 FP8 tensor rate is 2× BF16. Non-routed share (attn + indexer + shared + dense +
lm_head) ≈ 53% of FLOPs → ceiling **+25%**. Runtime activation-quantized scaled-mm for
`o_proj`/`q/kv_b_proj`/shared expert/down, with weights kept bf16. Quality gate: rerun the
malaiwah-style KLD panel; accept only ≤ +0.005 nats vs 0.0246 baseline. Do **not** touch
routed experts (trellis is the whole point of the 4bpw checkpoint).

### P7 — Serving-level aggregate wins (1–2 days, low risk)

- **Pack a second prefill into an idle step:** under `skip`, peer prefills fully defer.
  With MNBT≥3584 there is budget to co-schedule one small prefill alongside a long one's
  chunk when the long one's chunk is capped — improves aggregate prefill tok/s for the
  3×256k-style queue (today: 305/608/916 s).
- **UMA KV offload for evicted sessions:** the pool is 1.67M tokens; once it drains, an
  old session's turn is a full cold prefill. Copying a 100k-token MLA prefix back from a
  UMA slab is ~790 MB ≈ 4 ms at bus speed vs ~105 s of compute. A hot-prefix
  stage-out/stage-in (LMCache-style) converts repeat TTFTs into page copies. TTFT win,
  not tok/s — but it is the cheapest "prefill" speedup in the list.

### P8 — Drafter prefill trim (0.5 day, low priority)

DFlash2's drafter (5 SWA layers, block 8) prefills over the whole prompt. Its attention
is windowed, so for long prompts the drafter only needs target hidden states for its
window tail — restrict drafter prefill to the last W tokens and start drafting from
there. ~3% of step FLOPs; verify accept-rate unchanged on the structured/prose benches.

## 4. Expected outcome

| Stage | Cold 100k | Cold 300k TTFT | 8k TTFT |
|---|---:|---:|---:|
| Today | 947 tok/s | 323 s | 10.4 s |
| P1+P2 (chunk ladder + robust fused MoE) | ~1.2–1.4k | ~220–250 s | ~8–9 s |
| + P3 (if KDA confirms as wall) | ~1.5–1.9k | ~170–200 s | ~7–8 s |
| + P6 (fp8 non-routed) | ceiling ~2k+ | ~150–180 s | ~6–7 s |

1.3–2× realistic without touching weights or KV layout.

## 5. Do not (already priced or blocked on this kit)

- **MNBT=8192** — indexer smem (lab notes §5). Stop the ladder at 4096.
- **`--moe-backend marlin` / NVFP4** — KLD 0.0605 vs 0.0246, and SM12x NVFP4 kernels are
  dense-MHA only (no `fp8_ds_mla`).
- **`TRITON_ATTN`** — measured collapse (29 tok/s decode, accept 0.31).
- **TP=4** — lab notes §8; not ~2×, different topology, and this is a 2-node recipe.
- **Raising MNBT to "fix" APC** — different problem; APC hits are block-aligned and the
  overlay already handles the drafter SWA group. Re-verify APC after every P1 rung.
- **Dropping `MAX_MODEL_LEN`** to free memory — pool/concurrency math regresses.

## 6. Protocol

Reuse `docs/_run_cold_prefill.py` verbatim for every A/B (unique salt first, usage-based
`prompt_tokens`, one request at a time, metrics deltas for hits/compute). Add per-rung:
chunk count from logs, `vllm:time_to_first_token_seconds` histogram, and (once) a
profiler trace archived under `logs/prefill-profile-<date>/`. Every kept change needs:
health 200, `OK` content, no NaN, APC follow-up row ≥ current hit fraction, and the
structured/prose decode benches unchanged within noise (P1/P2 must not tax decode).

## 7. Open questions

1. What fraction of the 1.08 s chunk is GDN vs sparse-MLA/indexer vs MoE? (P0 answers.)
2. How often does the fat-expert fallback fire at MNBT=1024, and what is the max
   per-expert row count distribution? (One counter answers; informs P2 option choice.)
3. Does `exl3_moe` sustain >40% MFU at M≈227 rows/expert (3584-chunk), or does the
   trellis kernel fall over at large per-expert M? (Microbench: single layer, synthetic
   routing, sweep M — 30 minutes, no serve restart needed.)
4. Is there a dense path inside `FLASHINFER_MLA_SPARSE_SM120` for ctx ≤ topk? (Read
   `flashinfer/mla/_sparse_mla_sm120.py` in-image; P4.)
5. Does the second CX7 port pair light up cleanly (cable + `ibstat` + NCCL 2-rail ring)?

---

## 8. P0 results (2026-08-29) — do P2 before P1

Image `glm53-flash-sm121:prefill-p0`, MNBT=1024, `EXL3_MOE_ROW_TILE=0` (fallback still on).
Traces: `logs/prefill-profile-20260829/`. Torch profiler: delay 2 / max 3 engine steps.

### Q2 — fat-expert fallback (counter in `apply_exl3_fused_moe`)

Logged every 42 MoE layers (one engine step). Hist buckets: ≤16/32/64/128/256/512/1024/2048/+.

| After | MoE layers | fat_layers | max_rows | avg_max_rows | hist (le128 / 129–256 / 257–512 / 513–1024) |
|---|---:|---:|---:|---:|---|
| ~16k cold | 840 | **766 (91%)** | **1024** | 799 | 74 / 34 / 49 / 683 |
| ~100k cold | 5208 | **5134 (98.6%)** | **1024** | **920** | 74 / 34 / 57 / 5043 |

The 74 `≤128` layers are boot warmup/decode and stay frozen. On a 1024-token prefill chunk the hottest expert is in top-8 for **almost every token** (max_rows=1024, avg_max≈920). Fallback fires on essentially every prefill MoE layer. ~10.5 fat experts/layer (`fat_expert_slots / fat_layers`).

**Keep P2 option 2 (GPU 128-row tiles).** A TEMP_ROWS bump to 1024 would also cover today's max, but tiling stays bounded at 3584–4096. Option 3 (dequant grouped GEMM) not justified yet — fused `exl3_moe` still runs for the thin experts; the tax is the LinearEXL3 reconstruct loop + `aten::nonzero` host sync.

### Q1 — 1.08 s/chunk attribution (100k later-chunk capture, 3 steps, Self CUDA 2.96 s)

`execute_context` CUDA total **1.086 s/step** — matches the cold-ladder 1.08 s. 16k capture was slower (1.44 s/step) because profiler overlapped the first chunks.

| Wall (CUDA, 3 later 1024-token chunks) | Time | Share of 2.96 s | Per step |
|---|---:|---:|---:|
| `vllm::moe_forward_shared` (CUDA total, incl. children) | 1.87 s | **63%** | **623 ms** |
| `exl3_moe_kernel` fused | 401 ms | 14% | 134 ms |
| `reconstruct_kernel` (LinearEXL3 fat path) | 191 ms | 6.4% | 64 ms |
| `aten::index_add_` fat scatter | 156 ms | 5.3% | 52 ms |
| `aten::mm` (shared/dense/lm_head) | 307 ms | 10% | 102 ms |
| NCCL AR (`AllReduce` bf16 RING_LL) | 213 ms | 7.2% | 71 ms |
| sparse MLA prefill kernel | 128 ms | 4.3% | 43 ms |
| KDA/GDN (`chunk_gla` + `chunk_kda_*` + conv) | ~190 ms | ~6% | **~63 ms** |

CPU: `aten::nonzero` **1.45 s** in the 16k 3-step capture (29% CPU) — the fat `.nonzero().tolist()` host sync. KDA and indexer do **not** dominate; they stay flat 16k→100k (sparse MLA ~43 ms/step either rung).

### Profiled tok/s (not A/B — profiler + fat-log syncs on)

| Rung | TTFT | tok/s | gen |
|---|---:|---:|---|
| ~16k | 49.4 s | 324 | `OK` |
| ~100k | 126.2 s | 792 | `OK` |

Baseline unprofiled 16k/100k: 893 / 947 tok/s. Hits 0 on both.

### P2a — row-tile **revert** (MNBT=1024, `EXL3_MOE_ROW_TILE=1`, no profiler)

8 full-grid `exl3_moe` launches/layer (max_rows=1024 / 128) lost to the 128-row LinearEXL3 fallback:

| Rung | Baseline tok/s | Tile tok/s | Δ |
|---|---:|---:|---:|
| ~8k | 772 | **300** | −61% |
| ~16k | 893 | **354** | −60% |
| ~100k | 947 | **774** | −18% |
| ~8k APC | 7168 hits / 1.30 s | 7168 hits / 2.83 s | hits OK, slower |

Keep tile code behind `EXL3_MOE_ROW_TILE=0`. **P2b = bump `TEMP_ROWS_FUSED` to 1024** (one fused launch; hottest expert fits). Receipt: `logs/prefill-profile-20260829/p2-tile-mnbt1024.json`.

### P2b — temp-rows 1024 **revert** (MNBT=1024, `EXL3_TEMP_ROWS_FUSED=1024`, tile off)

One `exl3_moe` launch/layer (tokens ≤ 1024 so no fallback). Fat-log `hist_gt128` stayed empty — overflow path never ran. gen=`OK`, APC 7168/8004. Slower than 128+fallback on every rung:

| Rung | Baseline tok/s | 1024-row tok/s | Δ |
|---|---:|---:|---:|
| ~8k | 772 / 10.4 s | **672** / 11.9 s | −13% |
| ~16k | 893 | **704** | −21% |
| ~100k | 947 / 105.6 s | **721** / 138.8 s | −24% |
| ~8k APC | 7168 hits / 1.30 s | 7168 hits / 1.53 s | hits OK, slower |

Keep `TEMP_ROWS_FUSED=128` default. Env override stays for later MNBT probes. Receipt: `logs/prefill-profile-20260829/p2-temprows1024-mnbt1024.json`.

P2a tile and P2b 1024-row bump both lost at today's chunk. P1 ladder runs on 128-row fused + LinearEXL3 fat fallback.

### P1 — MNBT=2048 **keep** (TEMP_ROWS=128, tile off, no profiler)

Fewer chunks beat baseline even though fat fallback is hotter (max_rows=2048, ~98% layers fat, avg_max≈1830). gen=`OK`, APC 7168/8004. Coarser decode interleave.

| Rung | Baseline tok/s | MNBT=2048 | Δ |
|---|---:|---:|---:|
| ~8k | 772 / 10.4 s | **895** / 8.93 s | +16% |
| ~16k | 893 | **953** | +7% |
| ~100k | 947 / 105.6 s | **975** / 102.5 s | +3% |
| ~8k APC | 7168 hits / 1.30 s | 7168 hits / 1.27 s | hits OK |

Receipt: `logs/prefill-profile-20260829/p1-mnbt2048.json`. Next rung: 3584 (page-aligned 4×896).

### P1 — MNBT=3584 **revert** (TEMP_ROWS=128, tile off)

Page-aligned 4×896. gen=`OK`, APC 7168/8004, no OOM/NaN. Fat fallback hotter (max_rows=3584, ~96% layers fat, avg_max≈3050). LinearEXL3 overflow tax beats the fewer-chunks win vs MNBT=2048:

| Rung | Baseline | MNBT=2048 | MNBT=3584 | Δ vs 2048 |
|---|---:|---:|---:|---:|
| ~8k | 772 / 10.4 s | 895 / 8.93 s | **777** / 10.29 s | −13% |
| ~16k | 893 | 953 | **950** | ~0% |
| ~100k | 947 / 105.6 s | 975 / 102.5 s | **929** / 107.6 s | −5% |
| ~8k APC | 7168 hits / 1.30 s | 7168 / 1.27 s | 7168 / 1.25 s | hits OK |

100k is also slightly *below* the 1024 baseline. Do not keep 3584. Next rung: 4096 (ladder stop). Receipt: `logs/prefill-profile-20260829/p1-mnbt3584.json`.

### P1 — MNBT=4096 **revert** (TEMP_ROWS=128, tile off; ladder stop)

gen=`OK`, APC 7168/8004, no OOM/NaN. Fat fallback max_rows=4096, ~95% layers fat, avg_max≈3580. 100k is a hair above MNBT=2048; 8k and 16k lose, and 8k is also below the 1024 baseline. Coarser decode interleave is not worth the 8k TTFT regression.

| Rung | Baseline | MNBT=2048 | MNBT=4096 | Δ vs 2048 |
|---|---:|---:|---:|---:|
| ~8k | 772 / 10.4 s | 895 / 8.93 s | **755** / 10.59 s | −16% |
| ~16k | 893 | 953 | **948** | −1% |
| ~100k | 947 / 105.6 s | 975 / 102.5 s | **987** / 101.3 s | +1% |
| ~8k APC | 7168 hits / 1.30 s | 7168 / 1.27 s | 7168 / 1.30 s | hits OK |

**P1 winner: MNBT=2048** (tile off, TEMP_ROWS=128). 3584 and 4096 stay off. Never 8192. Production default baked to 2048. Receipt: `logs/prefill-profile-20260829/p1-mnbt4096.json`.

P1 ladder keep/revert:

| MNBT | 8k | 16k | 100k | APC | Decision |
|---|---:|---:|---:|---|---|
| 1024 (baseline) | 772 / 10.4 s | 893 | 947 / 105.6 s | 7168 / 1.30 s | baseline |
| 2048 | **895** / 8.93 s | **953** | 975 / 102.5 s | 7168 / 1.27 s | **keep** |
| 3584 | 777 / 10.29 s | 950 | 929 / 107.6 s | 7168 / 1.25 s | revert |
| 4096 | 755 / 10.59 s | 948 | **987** / 101.3 s | 7168 / 1.30 s | revert |

P2a/P2b stay reverted. Do not start P3–P8.

### P1 winner confirmation — MNBT=2048 production (decode + full ladder)

After baking `MAX_NUM_BATCHED_TOKENS=2048` (`start.sh` / `.env` / `.env.example`), restarted `IMAGE=glm53-flash-sm121:prefill-p0` with tile off, TEMP_ROWS=128, `GLM53_MIXED_PREFILL_CHUNK=skip`, `MAX_MODEL_LEN=1000000`. `/health` 200 empty body. Decode first (idle KV), then the full cold ladder. Chunks at 2048 are half the 1024 count (8k 4 vs 8; 100k 49 vs 98; 300k 147 vs 293). Trade: cold-prefill tok/s up, concurrent-decode interleave coarser.

Decode 5×400, thinking off, temp 0. Lab notes C1: structured 61.7 / accept 0.918; prose 26.9 / 0.332. No regression (structured slightly faster; prose within noise). No NaN.

| Bench | Lab tok/s | MNBT=2048 median | Accept | NaN |
|---|---:|---:|---:|---|
| structured | 61.7 | **65.9** | 0.975 | no |
| prose | 26.9 | **26.2** | 0.323 | no; coherent |

Receipts: `logs/prefill-profile-20260829/decode-structured-mnbt2048.json`, `decode-prose-mnbt2048.json`.

Full ladder (same harness as `docs/cold-prefill.md`). 8k here followed the decode benches, so it is slower than the idle P1 keep A/B (895 / 8.93 s). Long rungs still beat the 1024 baseline. APC 7168/8004.

| Rung | Baseline tok/s / TTFT | MNBT=2048 confirm | Δ tok/s |
|---|---:|---:|---:|
| ~8k | 772 / 10.4 s | **797** / 10.03 s | +3% (idle keep was **895** / 8.93 s) |
| ~12k | 896 / 13.4 s | **926** / 12.96 s | +3% |
| ~16k | 893 / 17.9 s | **958** / 16.69 s | +7% |
| ~100k | 947 / 105.6 s | **984** / 101.7 s | +4% |
| ~256k | 936 / 273.4 s | **973** / 263.2 s | +4% |
| ~300k | 928 / 323.2 s | **941** / 318.9 s | +1% |
| ~8k APC | 7168 / 1.30 s | 7168 / 1.28 s | hits OK |

Receipt: `logs/prefill-profile-20260829/p1-mnbt2048-full-ladder.json`. Aspirational P1+P2 ceiling (100k 1.2–1.4k, 300k 220–250 s) was **not** reached: P2a/P2b lost, so larger chunks could not drop the LinearEXL3 fat tax. Kept 2048 anyway because every confirm rung beats the 2026-08-29 ladder without decode tax.

Live production: MNBT=2048, TEMP_ROWS=128, tile off, mixed-prefill skip, num_active=-1. Stop after P2.
