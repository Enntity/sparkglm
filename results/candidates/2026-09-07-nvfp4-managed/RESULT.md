# Native NVFP4 passed the bounded operator and warmed TP2 fixture gates

The current compressed-tensors lane uses the pinned Red Hat checkpoint
240131d6a447c8d89acd428c5ddfc85598651744. The operator probe executes native
SM120 block-scaled E2M1 tensor-core kernels on GB10; this is execution evidence,
not just a device capability check.

The independent scalar W4A4 reference decodes the packed FP4 values and FP8
block scales, reproduces clamped SiLU multiplication, and requantizes the
intermediate activation. Relative L2 error was 0.00166–0.00168 across M=1,
17, 64, 65 and 257, with H4096/I1024, top-8 and 8 or 16 experts. The declared
limit was 0.03. The separate error against unquantized BF16 is about 0.266;
do not confuse quantization loss with error against the quantized algorithm.
Changing the clamp had a substantial effect. Changed-input CUDA graph replay
passed, and M65 CUDA memcheck reported zero errors. Profiler collection is
empty under memcheck; native kernel names are retained from unsanitized runs.

The earlier v2 reference incorrectly interpreted FP8 scale bytes as integer
magnitudes and failed. That receipt remains retained; source commit 6cfa8f9
corrects the reference, and b9c28fa adds the routing/graph cases. No declared
numerical threshold was relaxed.

LLooM started the separate weightless TP2 fixture with native FlashInfer
CUTLASS. Its first two-repetition warmup had a mixed-C4 token mismatch. The
subsequent warmed three-repetition run completed every stream and preserved
signatures within all cases: C1 153.2 output tokens/s, mixed C4 368.7, long C2
78.1. These synthetic weights differ from the EXL3 fixture, so these numbers
are not a cross-format performance or quality comparison. Warmup/batch-shape
nondeterminism remains a limitation.

The fixture image initializes packed bytes, positive scales and normalization
weights only behind the synthetic-fixture marker and dedicated environment
flag. Real model weights are never initialized this way. Both model runtimes
were started/stopped by LLooM; full-model performance and quality remain
outstanding at this qualification level. This bounded operator screen does
not exhaust production expert-count shapes or long-context behavior.

The optional MXFP8 drafter is separately pinned to
610aa967a92bfeb97e3d848dcb8693553e8b6a55. Its six files and published weight hash
were independently verified on both ranks. Preparation alone does not qualify
MXFP8 inference, Humming, draft TP1, or any new performance setting.
