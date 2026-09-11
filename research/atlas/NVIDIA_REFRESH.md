# NVIDIA ModelOpt Atlas experiment

This disabled AGPL research patch refreshes the Atlas experiment from
[Mango-kid/atlas](https://github.com/Mango-kid/atlas) revision
`90b3584abc71b44b609637092b85d8423d8ff20f`. It does not change SparkGLM's
vLLM default, and no serving/performance qualification is implied.

The upstream implementation supplies native NVFP4 execution, native GLM MTP,
and its recent ownership/serving work. The local patch adds:

- direct loading and validation of NVIDIA's packed initial dense FFN weights
  and scales, avoiding an unnecessary dequantize/requantize roundtrip;
- explicit `ATLAS_GLM_SSM_NORMALIZE=0` for the GLM family, preserving the
  original SparkGLM Atlas contract without the generic recurrent-state clamp;
- omission of the unused appended predictor layer when GLM speculation is off,
  consistently across both loaders and their byte estimates;
- flushing held reasoning at stream completion into its original channel;
- focused regression tests using actual loader and model-dispatch paths.

Unset/1 normalization retains the fork's existing behavior. Other model
families are unchanged. The normalization opt-out is a semantic experiment,
not a demonstrated explanation for the fork's reported answer failures.
Inherited Atlas arithmetic still quantizes some checkpoint BF16 components;
this is not a claim of NVIDIA/vLLM numerical parity.

## Reconstruction

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Mango-kid/atlas.git atlas-nvidia
cd atlas-nvidia
git checkout 90b3584abc71b44b609637092b85d8423d8ff20f
git apply /path/to/sparkglm/research/atlas/nvidia-modelopt-refresh.patch
```

All source remains AGPL-3.0-only. The patch adapts the pinned upstream loader
and dispatch; the regression cases and explicit opt-out are original work,
informed by SparkGLM's archived `775cb3655e29a3735f4f58faa540608f9427bf51`
GLM recurrence contract. Preserve upstream license headers and notices.

## Initial validation

- NVIDIA checkpoint pin: `423acf37583782c51c142d145aef733d72943d93`.
- Actual metadata from all 33 shards inspected; all nine initial dense FFN
  projections satisfy the new packed-weight contract.
- 19 loader tests and four normalization dispatch tests pass on both the Mac
  CPU path and the native aarch64 CPU path.
- Rust 1.93.1 / CUDA 13.0 native release build succeeds, including 212 kernels.
- NCCL is pinned to 2.31.2 (CUDA 13.3 package); CUTLASS is pinned to
  `cf064d2e6bad2886238ac565b3b49007764f4939`.
- Both packaged ranks have ELF SHA256
  `be8835b87ea31a4f68da0d7c659eb35003abbc179f30161e6745f4e1f6403b3a`.
- Runtime image ID:
  `sha256:b7a3231b47ea91142a7483f222105fcff20659b3848a4dc8d020175be064cda4`.

The first image completed a separate four-layer fixture and full-model C1
synthetic chat, JSON, tool roundtrip, retrieval and cancellation probes. These
are bounded bring-up checks, not a complete integration or semantic gate.
Quality issues included unrelated reasoning, premature text alongside a tool
call, and a short response ending in reasoning without final-answer content.
The stream flush fix preserves that reasoning; it does not reclassify it as a
final answer or claim to fix the model's answer-quality failure.

Native MTP's first attempt was refused by the loader's memory preflight before
rank-0 tensor allocation. NVIDIA's predictor tensors are present, but their
BF16 upload peak requires separate loading work. The memory guard remains.

Final source: `561ffcbf59c12982103984dbbc0060b867b6c315` in the private engine
experiment, represented completely by this patch against the upstream pin.
Patch SHA256: `c8e96fba76bc9a35151df6f9d0a035a492c08d1255593469ef4b741ad434cf89`.
The combined native build passes 68 focused tests: 19 model-loader, four
normalization, three physical-loader, one configuration-policy and 41 streaming
tests. The streaming omission has a native red-to-green reproduction.

Final ELF SHA256:
`804dd9c9be4dc682e1802fae6a1033b96c07c4aeda11f10335b2a754eca73581`.
Final runtime image ID:
`sha256:0ac163a69075b248c1817466ee6854996c71873e08a8df74f95c9973a7f8ca96`.
Final full-model revalidation confirms ordinary serving and the streaming
repair. The repeated short chat/tool content remains correct where initially
correct; a longer retrieval case ends with its answer only in reasoning.
That case passed initially, so token-equivalence and broad quality remain
unqualified. The unused-predictor omission reduces reported uploaded bytes as
predicted. No throughput advantage, long-context, MTP, endurance or
default-promotion claim is made here.

The checksum-bound [bring-up record](../../results/candidates/2026-09-10-atlas-nvidia-bringup/RESULT.md)
retains the selected synthetic outputs and the failed qualification boundaries.
