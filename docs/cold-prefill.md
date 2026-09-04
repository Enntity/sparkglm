# Cold prefill — live remeasure (2026-08-29)

Live cold-prefill ladder on the **GLM-5.3-Flash EXL3** 2× DGX Spark serve. Serve flags were not changed. The server was not restarted.

| | |
|---|---|
| Endpoint | `http://127.0.0.1:8888` |
| `GET /health` | **200** (empty body) before and after |
| Served id | **`GLM-5.3-Flash-EXL3`** (`GET /v1/models`) |
| `max_model_len` | **1,000,000** |
| KV pool (metrics) | **1,691,099** tokens, `cache_dtype=fp8`, prefix caching on, `block_size=64` |
| When | 2026-08-29, UTC afternoon |

Harness: `docs/_run_cold_prefill.py`. Raw JSON: `docs/_cold_prefill_raw.json`.

## Protocol (matches the published kit receipts)

- `temperature=0`
- thinking **off** via top-level `"chat_template_kwargs": {"enable_thinking": false}` (not `extra_body`)
- `stream=true` with `stream_options.include_usage=true`
- `max_tokens=8` (prefill / TTFT, not decode)
- unique salt (UUID + random pad) **first** in the user message so prefix cache cannot cheat
- filler `"the "` + `"Reply with OK."`; `/tokenize` used only to pick filler count
- one request at a time; each finished before the next started
- **TTFT** = wall time from request start to first **content** token
- **prompt_tokens** from the stream `usage` object (not an estimate)
- **prefill tok/s** = `prompt_tokens / TTFT`
- APC after ~8k: **same** user text + assistant reply + short follow-up user turn (`Confirm with OK.`)

Every cold rung reported **0** prefix-cache hits. Usage `prompt_tokens` was within 0.1% of each target (well inside ~2%).

## New vs old kit receipts

Prefill tok/s is the comparable column. Old ~8k / APC used ~7.7k prompt tokens; this run targeted 8k.

| Rung | Old TTFT / tok/s | New prompt tok | New TTFT | New tok/s | Δ tok/s |
|---|---|---:|---:|---:|---:|
| ~8k cold | 9.7 s / **793** | 7995 | 10.36 s | **772** | −2.6% |
| ~12k cold | 13.4 s / **895** | 11995 | 13.38 s | **896** | +0.1% |
| ~16k cold | 17.7 s / **904** | 15995 | 17.91 s | **893** | −1.2% |
| ~100k cold | **934** tok/s | 99995 | 105.56 s | **947** | +1.4% |
| ~256k cold | 305 s / **~839** | 255995 | 273.44 s | **936** | +11.6% |
| ~300k cold | 356 s / **~840** | 299995 | 323.23 s | **928** | +10.5% |
| ~8k follow-up (APC) | 1.17 s, **7168 / 7717** hits | 8004 | **1.30 s** | — | **7168 / 8004** hits |

Short rungs sit on the old receipts (within a couple percent). Long rungs are faster: ~256k and ~300k moved from ~840 tok/s / 5-minute-class TTFT to ~930 tok/s.

APC still hits the same **7168**-token block-aligned prefix (2 × 3584 hybrid align). This run’s prompt is longer (8004 vs 7717), so the uncached tail is **836** compute tokens vs the old **549**, which accounts for 1.30 s vs 1.17 s TTFT. Hit fraction is 7168/8004 = **89.6%** (old 7168/7717 = 92.9%).

## Per-rung record

| Rung | HTTP | finish | prompt_tokens | completion_tokens | TTFT s | prefill tok/s | hits / compute | gen |
|---|---:|---|---:|---:|---:|---:|---|---|
| ~8k cold | 200 | stop | 7995 | 2 | 10.355 | 772.1 | 0 / 7995 | `OK` |
| ~8k follow-up | 200 | stop | 8004 | 2 | 1.298 | 6167.4 | **7168 / 836** | `OK` |
| ~12k cold | 200 | stop | 11995 | 2 | 13.382 | 896.3 | 0 / 11995 | `OK` |
| ~16k cold | 200 | stop | 15995 | 2 | 17.911 | 893.0 | 0 / 15995 | `OK` |
| ~100k cold | 200 | stop | 99995 | 2 | 105.557 | 947.3 | 0 / 99995 | `OK` |
| ~256k cold | 200 | stop | 255995 | 2 | 273.443 | 936.2 | 0 / 255995 | `OK` |
| ~300k cold | 200 | stop | 299995 | 2 | 323.230 | 928.1 | 0 / 299995 | `OK` |

Hits / compute are deltas of `vllm:prefix_cache_hits_total` and `vllm:prompt_tokens_by_source_total{source="local_compute"}` across that request. `usage.prompt_tokens_details.cached_tokens` was not populated (`--enable-prompt-tokens-details` is not required for this bench).

All seven requests: HTTP 200, `finish_reason=stop`, completion 2 tokens, content exactly `OK`, empty reasoning (thinking stayed off), no `nan`.

## Metrics sanity

Start: queries 32668, hits 14336, prompt_tokens 32668, local_compute 18332.

End: queries 732642, hits 21504, prompt_tokens 732642, local_compute 711138.

Deltas: **+699974** queries / prompt tokens, **+7168** hits, **+692806** local compute. Sum of this ladder’s `usage.prompt_tokens` is 7995+8004+11995+15995+99995+255995+299995 = **699974**. Hits increased by exactly the APC 7168. `GET /health` still 200; `num_requests_running=0`.
