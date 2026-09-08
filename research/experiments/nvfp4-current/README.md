# Current NVFP4 candidate

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
symbols before making a hardware acceleration claim. Mixed FP8 layers require
per-layer automatic backend selection; do not globally force an incompatible
MoE backend or remove the clamp guard to obtain a successful load.

The initial bring-up profile uses 8 GiB KV per rank and 262144 context because
this checkpoint is larger than EXL3. Final comparison must repeat EXL3 with
these same cache limits, or explicitly separate capacity from speed. Use the
frozen exact-token C1/C2/C4 matrix with the same prompt salts, output lengths,
three paired repetitions and discarded warmup. Protect tools, reasoning,
prefix-hit, cancellation and quality. No winner is established by preparation,
backend availability, or a single successful request.

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
