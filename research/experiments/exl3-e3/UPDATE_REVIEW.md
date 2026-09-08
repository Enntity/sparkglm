# September 7 update review

The selected EXL3 experiment is the complete Mia E3 arithmetic schedule at
MIT revision `2c0ebe55a91ac8c0868cd6cba264e14bce93c66b`, including the SiLU
precision correction. The upstream comparison was against its E2 host loop;
SparkGLM already has grouped M64 prefill and cooperative K4 decode, so that
reported speedup is not transferable. Our operator screens rejected universal
dispatch and the 2048-token threshold. The current opt-in adapter starts at
4096 input tokens and keeps the existing decode path. See the checksummed
screen and managed-integration records under `results/candidates`.

The pinned kernel, builder and extracted tables retain MIT attribution. Later
upstream licensing changes do not alter the identity of the copied source;
do not refresh these files from a floating branch. The source-locked installer
and local dispatch policy are separate original changes.

The current NVFP4 experiment uses the September 7
[Red Hat compressed-tensors checkpoint](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4/tree/240131d6a447c8d89acd428c5ddfc85598651744).
It is a different format from the earlier prepared ModelOpt/Humming recipe.
The implementation and qualification live on `experiment/nvfp4-tp2`, under
`research/experiments/nvfp4-current`. Routed expert layers 3–44 use group-16
W4A4; layer 45 uses FP8. Automatic per-layer backend selection is necessary:
forcing one incompatible backend across the mixed checkpoint is not a valid
way to demonstrate native FP4. Keep SwiGLU clamping and verify actual GPU
kernel symbols, numerical behavior and real-model quality separately.

Other upstream changes require their own hypotheses. Compact NoPE cache,
decode context parallelism, an MXFP8 drafter, and replacing DFlash2 with MTP
all change the comparison beyond the routed-expert arithmetic. They are not
silently included in this experiment. Clock caps and library version bumps
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
