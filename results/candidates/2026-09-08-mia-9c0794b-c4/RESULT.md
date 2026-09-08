# Mia 9c0794b as-shipped C4 comparison

The upstream default recipe completed all 12 requests without API errors or preemptions. Median elapsed time was **128.1759 s**, versus the archived current SparkGLM NVFP4 median **83.868935 s**. For 1,600 output tokens, that is **12.48 vs 19.08 output tok/s including all waiting**. NVFP4 throughput was 1.53x on this workload; configurations differ.

Runs: 135.489874, 128.1759, 123.534565 seconds. Median selection uses full elapsed wall time; no timing is trimmed from the video. Median TTFT by arrival: 19.421603, 54.027566, 82.655923, 103.029816 seconds. Decode-only per stream: 18.820258, 22.534739, 23.351263, 18.016817 tok/s.

Four unique approximately 16K prompts arrive at 0/1/2/3 seconds, each requesting 400 output tokens, temperature zero and thinking disabled. Exact input counts 15,807 / 15,810 / 15,810 / 15,809 match the earlier NVFP4 capture. Cache salt changes per run, isolating prefix-cache reuse without changing prompt text.

Pinned source: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/9c0794b68d7fc124f79104409ab434769503fb31 . Unmodified Dockerfile built image f301bb941416d40a0080fb2505a275caf7c71eb4d94471dd4539f0d0b2edbde2, present on both ranks. The mutable published image differed from this source overlay and was not used for the measurement.

Preserved defaults: E3 grouped, temp rows 32, skip mixed-prefill policy, stock spinwait, rightsize indexer, 850,000 context, 0.85 memory utilization, 7,168 prefill chunk, four sequences, DFlash2 k7 TP2, BF16 drafter, FP8 target KV, vision enabled, graph estimation enabled. Adaptive-k and dense-FP8 remain off. LLooM owns both ranks and the measured gateway model `mia-glm53-9c0794b`; per-rank launch scripts were extracted unchanged from upstream. Host fabric, local paths and cache locations were mapped for this installation.

The 120 target shards (175,642,157,752 bytes) were reused as local hard links on both hosts. Target revision 25a44fdbf16862a46b7cc9921142c6c81350af2f. The current draft revision bf582e4eacc1810f76656d1811693ff6c6737d2a changes model.safetensors relative to the older local draft; it was acquired separately and its SHA256 b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac matched on both hosts.

Engine startup reported 14.82 GiB available KV per rank and 929,887 cache tokens, admitting the configured 850K context. Observed available host memory during C4 was around 6–8 GiB. Preemption counter was zero before and after all captures. This does not establish safety or speed for a full 850K prompt.

The upstream boot-shape script passed 20/20 directly against the LLooM-managed backend. Initial gateway warmups reached generation but could not call the unexposed `/tokenize` endpoint; they are setup, outside timing. All measured requests used the authenticated LLooM gateway and route receipts show the new runtime without fallback. No broad semantic or tool-quality claims are made.

The visible serial waiting is consistent with the preserved skip policy. This comparison does not isolate that policy from kernels, quantization, draft weights or context allocation. It is evidence for this as-shipped recipe on the specified staggered workload, not a general EXL3 verdict.

Video: generated outside the repository under sparkglm-artifacts; its checksum and media properties are recorded in video.json. Real SSE text arrival timing is retained at 1x. Animated token counts distribute exact final usage over text deltas.
