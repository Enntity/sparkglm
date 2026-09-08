> Historical September 7 campaign notes. The September 8 public install now
> selects the same measured serving source at 512K/9 GiB; see
> [the current default](../../../SPARKGLM.md) and
> [capacity evidence](../../../results/candidates/2026-09-08-nvfp4-context-512k9/RESULT.md).
> Earlier bring-up descriptors below remain historical experiments.

# Current NVFP4 candidate

The final measured configuration for this campaign is native W4A4 with the
MXFP8 DFlash2 draft, actual target/draft TP2, seven draft tokens, 2048-token
prefill chunks, 65536 context, and 9 GiB KV per rank. It has three retained
full matrices; see [the result](../../../results/candidates/2026-09-07-nvfp4-native-2k/RESULT.md)
and [the complete comparison](../../../results/candidates/2026-09-07-tp2-comparison/RESULT.md).
It retains a bounded arithmetic failure and is not a general-quality or
release-qualified default. The bring-up notes below preserve earlier trials.

`Dockerfile.mxfp8` builds the pinned draft support from this branch over the
verified reference image. Preserve both immutable rank image IDs and run the
recorded source/operator/integration gates for a new build. In the LLooM
checkout, materialize the selected settings with `--nvfp4 --mxfp8-draft
--draft-tp 2 --prefill-tokens 2048 --context-tokens 65536 --kv-cache-gib 9`
and the verified image IDs/source revision. LLooM owns the resulting runtime.
Stop the active appliance runtime before replacing its managed profile.

Unqualified experiment on branch `experiment/nvfp4-tp2`. This second lane uses
Red Hat's September 7 compressed-tensors checkpoint at
`240131d6a447c8d89acd428c5ddfc85598651744`, whose routed expert layers 3–44 use
NVFP4 W4A4; layer 45 uses FP8. The prior copied ModelOpt/Humming experiment in
`../nvfp4-tp2` remains separately pinned and unqualified. Never swap their
checkpoint revisions: the formats differ.

Reuse the preserved SparkGLM reference image, including the same DFlash2 draft,
TP2 topology, seven draft tokens, scheduler, parser, and graph settings. LLooM's
`backends/sparkglm/materialize.mjs --nvfp4` produces an immutable-image managed
runtime with its own model identity and no production aliases. LLooM owns
worker-first start, readiness, admission, stop, and gateway routing. Building
an image separately does not make its process externally managed.

The preserved image reports SM121 and FlashInfer CUTLASS availability. This
is a capability check only: record actual per-layer selection, verify SwiGLU
clamping, run a kernel correctness check, and capture native FP4 profiler
symbols before making a hardware acceleration claim. Backend overrides must pass per-layer compatibility validation; never remove
the clamp guard to obtain a successful load. Layer 45 is next-token prediction
metadata and is not evidence of an FP8 MoE in the DFlash2 target forward.

The initial 8 GiB KV / 262144-context profile loaded the target but failed
vLLM cache admission: 9.31 GiB was required and its estimated maximum length
was 78848. The next bring-up uses 65536 context with the same 8 GiB budget;
this is a configuration correction, not evidence of an FP4 kernel failure.
LLooM exposes explicit context, KV, prefill and backend controls. Draft TP
must remain 2 with the inherited loader.
Matched profiles isolate effects diagnostically. Final selection compares each
path's fastest working TP2 configuration and reports its context capability.
Use the frozen exact-token C1/C2/C4 matrix with prompt salts, output lengths,
three repetitions and discarded warmup. Protect tools, reasoning, prefix-hit,
cancellation and quality. Preparation or one successful request is no winner.

Sources: [current checkpoint](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/tree/240131d6a447c8d89acd428c5ddfc85598651744).
The descriptor and this integration design are original Apache-2.0; no model
weights or upstream implementation are copied here.

The weightless NVFP4 fixture uses the same tinyGLM geometry and a pinned copy of
the current quantization metadata. Its isolated image initializes packed bytes
and positive scales deterministically; stock vLLM leaves integer dummy weights
uninitialized. The hook is gated by a dedicated environment variable and checks
the fixture marker before touching tensors. It must never be used with real
weights. EXL3 and NVFP4 dummy quantization differ, so cross-format token equality
is not a quality contract; record repeatability plus the quantized operator
reference and full-model semantic comparison.

The first MXFP8 / 16K-chunk attempt loaded the target and selected
FlashInferCutlassMxfp8LinearKernel for the draft, then failed cache admission:
15.2 GiB was required for 64K context, against 9 GiB allocated. The padded
sliding-window draft cache charges its in-flight prefill budget, so increasing
chunk size substantially raises the reservation. The next screen uses 1K
chunks, 9 GiB KV, and 64K context.

The configured draft TP1 flag in that failed attempt did not change execution:
the installed DFlash loader retains the target parallel configuration. Its
2048-byte/token draft KV geometry matches TP2. LLooM now rejects unsupported
independent draft TP before loading weights. Historical copied TP1 profile
metadata is not evidence of a TP1 runtime on this base.
