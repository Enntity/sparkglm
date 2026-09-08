# September 7 update review

The selected EXL3 experiment is the complete Mia E3 arithmetic schedule at
MIT revision `2c0ebe55a91ac8c0868cd6cba264e14bce93c66b`, including the SiLU
precision correction. The upstream comparison was against its E2 host loop;
SparkGLM already has grouped M64 prefill and cooperative K4 decode, so that
reported speedup is not transferable. The initial operator screens rejected universal
dispatch and the 2048-token threshold at the original expert-row setting.
The later 32-row expert threshold has separate passing operator evidence from
2048 input tokens; other row thresholds retain the 4096 guard. The corrected
concurrent adapter reserves both scratch paths during profiling and keeps
solo prefill on the reference path. See the checksummed fixed-cache controls,
profiling correction, and full-model records under `results/candidates`.

The pinned kernel, builder and extracted tables retain MIT attribution. Later
upstream licensing changes do not alter the identity of the copied source;
do not refresh these files from a floating branch. The source-locked installer
and local dispatch policy are separate original changes.

The current NVFP4 experiment uses the September 7
[Red Hat compressed-tensors checkpoint](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/tree/240131d6a447c8d89acd428c5ddfc85598651744).
It is a different format from the earlier prepared ModelOpt/Humming recipe.
The implementation and qualification live on `experiment/nvfp4-tp2`, under
`research/experiments/nvfp4-current`. Routed expert layers 3–44 use group-16
W4A4; the FP8 layer 45 belongs to next-token prediction metadata and is not
executed by the DFlash2 target forward. It therefore does not require a mixed
FP8 MoE backend in this serving path. Preserve backend compatibility and
SwiGLU clamping, and verify actual GPU kernel symbols, numerical behavior and
real-model quality separately. Native W4A4 and Marlin W4A16 are distinct
complete arithmetic paths, even when they consume the same packed weights.

Other upstream changes require their own hypotheses. Compact NoPE cache,
decode context parallelism, and replacing DFlash2 with MTP
change the comparison beyond the routed-expert arithmetic. They are not
silently included in this experiment. MXFP8 draft support now has separate
explicit screens on both branches; it is not part of the pure EXL3/E3 image. Clock caps and library version bumps
also need a demonstrated gain against this appliance's existing baseline.

One correctness detail is especially easy to backport incorrectly:
[vLLM PR 55234](https://github.com/vllm-project/vllm/pull/55234) preserves an
explicit `AssertionError` when incompatible cache specs are merged under
optimized Python. Its callers use that exception to select a fallback;
changing just the assertion to `ValueError` breaks the caller contract.
It also restores non-causal MLA decoding as an aggregated capability.
The measured SparkGLM profile does not enable optimized Python, and its
DFlash2 SWA grouping has separate local patches. This is not included as an
unmeasured performance change or a blanket cache-group rewrite.

All image builds are separate from runtime ownership. LLooM's SparkGLM
adapter pins both rank images, starts workers first, applies ordinary
admission, and owns stop and gateway routing. An installed recipe or a
healthy endpoint does not promote the candidate: the documented G3/G4 and
release gates still apply.
