# X draft — NVFP4 update

SparkGLM has moved to NVFP4 as its default on two DGX Sparks. Our focus is
concurrent serving: getting several requests through prefill and keeping
agents moving when work overlaps.

Mia is doing great work on GLM on Sparks. Her two-Spark recipe is an important
foundation for ours, and we're grateful for it. We're pursuing a particular
tradeoff: better performance under concurrent load.

In our frozen three-run comparison against our original EXL3 configuration:
• C1 / 16K: time to first token 11.1 → 8.9 seconds.
• C1 / 32K: 21.5 → 17.1 seconds — about 20% less waiting.
• C4 / 16K: all four requests finished in 57.2 rather than 78.5 seconds —
  27% less wall time, or 37% more throughput for the same output workload.
Our updated EXL3 build landed at 67.3 seconds in that C4 test.

Why: native FlashInfer CUTLASS NVFP4 compute, an MXFP8 DFlash2 draft, 2K
prefill chunks and mixed scheduling. Those are complete tuned configurations;
this isn't a claim that quantization alone caused every gain. We kept the
kernel/operator checks, tinyGLM integration, full-model runs and failed
experiments in the public repo.

The tradeoff is memory. This NVFP4 checkpoint has a larger resident footprint
than our EXL3 quant: packed weights are only part of the cost; scales,
higher-precision components, the draft, KV and workspaces also occupy the
same unified memory. We selected 512K context with 9 GiB KV per rank after
our 1M attempt ran out of room. 512K passed a near-full-window prompt and
cached tool continuations. Headroom remains tight; it doesn't mean four simultaneous 512K requests.

For agents, prefix caching can make subsequent tool turns much faster — our
near-full-window cached turns were around two seconds — but contention and
preemption can still make waits long. Small quality probes also found misses;
we're not claiming broad quality equivalence or production-grade endurance.

The C1/C4 figures above used the earlier 64K NVFP4 profile versus 1M EXL3.
The two videos we're posting next use the newer 512K profile and a separate
staggered C4 workload: ours mixed vs Mia's default skip, then skip vs skip.
Both show SparkGLM on top. Context, drafts and settings are disclosed.

Latest NVFP4 is on main; latest EXL3 stays available on the exl3 branch.
Source, exact settings and results: https://github.com/Enntity/sparkglm

EXL3 checkpoint credit: Brandon M. Music's ShapleyMCG (2026), “ShapleyMCG:
An Auditable Calibration-to-Encoding Pipeline for Low-Bit Mixture-of-Experts
Models.” https://github.com/brandonmmusic-max/shapleymcg

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## Video captions

1. **SparkGLM NVFP4 mixed / Mia default skip.** Four staggered ~16K prompts,
400 output tokens each; median of three runs, 1x playback. 83.87s / 19.08 tok/s
vs 128.18s / 12.48 tok/s. These are whole-run output rates, including prefill
and waiting. 512K vs 850K configured context; different draft and chunk settings.

2. **SparkGLM NVFP4 skip / Mia default skip.** Same staggered workload and
capture method. 116.12s / 13.78 tok/s vs 128.18s / 12.48 tok/s. Our skip retains
its small-remainder bypass; scheduler implementations are not identical.

Both video posts also need the EXL3 checkpoint credit/notice above. The default
mixed run trades more preemption for better total completion time in this test.

## Evidence (editor notes, not post copy)

- [Frozen suite](../results/candidates/2026-09-07-tp2-comparison/RESULT.md)
- [512K capacity](../results/candidates/2026-09-08-nvfp4-context-512k9/RESULT.md)
- [Mixed video](../results/candidates/2026-09-08-nvfp4-c4-video/RESULT.md)
- [Skip video](../results/candidates/2026-09-08-nvfp4-skip-c4/RESULT.md)
- [Mia capture](../results/candidates/2026-09-08-mia-9c0794b-c4/RESULT.md)

This is long-post copy, not a 280-character post. No X publication has occurred.
