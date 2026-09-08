# NVFP4 skip C4 and stacked comparison videos

The current SparkGLM NVFP4 recipe with skip completed all 12 requests without API errors or preemptions. Runs took **117.416358, 116.122222, 110.048569 seconds**. The median run is r2, **116.122222 s**, or **13.778586 output tok/s including waiting** for 1,600 output tokens.

Mia 9c0794b as-shipped skip median is **128.1759 s / 12.482846 tok/s**. The archived SparkGLM NVFP4 mixed median is **83.868935 s / 19.077385 tok/s**. On this workload, ours with skip has about 10.4% higher end-to-end output throughput than Mia's skip recipe. Our mixed recipe remains faster than our skip variant. This is not isolated kernel evidence or a broad quality evaluation.

Median skip TTFT by arrival is **10.854077, 40.198384, 63.737527, 89.455751 s**. Four approximately 16K unique prompts arrive at 0/1/2/3 seconds and each requests 400 output tokens with temperature zero and thinking disabled. Prompt text and exact input token counts match the earlier captures. Cache salt isolates prefix reuse per run without changing prompt text.

Only the NVFP4 inference setting `GLM53_MIXED_PREFILL_CHUNK` changed from `0` to `skip`. The target and MXFP8 draft weights, native CUTLASS kernel, 512K context, 9 GiB KV per rank, 2048-token prefill chunk, four-sequence limit, DFlash2 k7 TP2, and 16ms spinwait were preserved. LLooM installed and managed the separate gateway model `sparkglm-nvfp4-skip`. Actual container environment was verified on both ranks; all measured calls went through LLooM. The original model's configuration was preserved for restoration.

Our installed scheduler also retains its existing 3584-token remaining-prefill bypass; max-wait stays zero. Therefore this is each implementation's skip mode, not byte-identical scheduling code. Mia additionally differs in context, prefill chunks, target/draft formats and other serving choices. A health/generation canary preceded the NVFP4 captures; initial model load and graph initialization are excluded from measurement. No additional tuning was performed.

The before/after preemption counters are zero. Available host memory was observed around 3.4–5.3 GiB on the head during this skip campaign; this is narrower than the earlier Mia campaign's observed headroom and does not establish long-context safety. No OOM or request errors occurred in these runs.

Both videos use the requested over-under layout with SparkGLM above Mia and four visible streams in each panel. The previous mixed-versus-skip video was re-rendered from the exact same stored captures. The new skip-versus-skip video uses median r2 for each implementation. Full 1x elapsed time plus a four-second final hold is retained; text timing is actual SSE arrival, while token counters distribute exact final usage across deltas. Generated media stay outside this repository; media properties and checksums are in videos.json.
