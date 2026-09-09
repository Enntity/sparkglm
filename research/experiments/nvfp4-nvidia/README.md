# NVIDIA ModelOpt NVFP4 checkpoint integration

This opt-in experiment uses NVIDIA's checkpoint at
`423acf37583782c51c142d145aef733d72943d93`. Red Hat remains the default.
The NVIDIA model card and configuration are available at
https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/423acf37583782c51c142d145aef733d72943d93.

The profile uses `modelopt_fp4`, native FlashInfer CUTLASS MoE, TP2,
FP8 KV and the existing MXFP8 DFlash2 draft. The checkpoint also quantizes
the initial dense MLPs. Its loader format is different from Red Hat's
compressed-tensors checkpoint. Do not point the default profile at these files.

Inspect the explicit profile before installing:

```sh
./start.sh --profile nvfp4-nvidia plan
./start.sh --lloom --profile nvfp4-nvidia plan
```

The standard installation flags in `SPARKGLM.md` apply, with
`--profile nvfp4-nvidia`. It has a separate runtime/model ID,
`sparkglm-nvfp4-nvidia`, and backend port 8892. It does not change aliases or
the default model. Use a separately verified image; the profile's presence
does not certify a rebuilt image or the configured 512K context limit.

## Qualification

- `candidate.json` pins the upstream configuration hash.
- `make_tiny.py` creates a three-layer, 16-expert synthetic model covering
  dense KDA, routed KDA and routed sparse MLA at production hidden dimensions.
- `tiny_init.py` initializes valid deterministic packed FP4 bytes and scales
  only for the marked synthetic model. Never enable it for real weights.
- `Dockerfile.tiny` adds the guarded initializer to an existing serving image.
- `tiny-entrypoint.sh` adapts the standard launcher for the synthetic marker
  and ModelOpt quantization. It is a test launcher, not a production adapter.
- `linear_probe.py` checks the ModelOpt dense linear path at TP2 dimensions
  against independently decoded FP4 operands, including changed-input CUDA
  graph replay. The existing `nvfp4-current/kernel_probe.py` checks clamped MoE.

Gate status and measured limitations must be recorded in a checksum-bound
result bundle before claiming full-model compatibility or performance.
NVIDIA's GB200 model-card quality measurements are not GB10 qualification.

The model is fetched separately under its publisher's terms. The existing
draft's separate non-commercial/no-derivatives terms still apply. No weights,
model-derived tensors, or vendor serving code are included here.
