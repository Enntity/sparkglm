# NVIDIA native MTP overlay converter

AGPL-3.0-only research source. The quantizer is copied unchanged from the pinned
Mango Atlas revision in `provenance.json`; wrapper, validation and tests are
original campaign work. No library, tensors, weights or machine orchestration
are distributed. Model publisher terms remain separate.

Use NVIDIA checkpoint revision `423acf37583782c51c142d145aef733d72943d93`.
The converter replaces exactly864 BF16 predictor expert matrices in shards1–3
with Atlas runtime-format NVFP4. It preserves and rechecks all other tensor
payloads. This reproduces Atlas's runtime quantization offline; it does not
reproduce NVIDIA calibration or vLLM's BF16 predictor numerics.

Build with CUDA13 on the target architecture:

```sh
nvcc -O3 -arch=sm_121f --fmad=false --expt-relaxed-constexpr -DTQ_PLUS_SIGNS \
  -shared -Xcompiler=-fPIC wrapper.cu -o libatlas_mtp_quantize.so
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -v
```

After stopping GPU serving, set `CHECKPOINT` to the absolute local snapshot path
and `OVERLAY` to a new directory. The output must not exist for conversion.

```sh
python3 convert.py --source "$CHECKPOINT" --output "$OVERLAY" --one-matrix
python3 convert.py --source "$CHECKPOINT" --output "$OVERLAY"
python3 convert.py --source "$CHECKPOINT" --output "$OVERLAY" --verify-overlay
```

The one-matrix canary checks determinism, geometry and scale range, not model
quality. Only `conversion.complete.json` marks completed payload verification.
Unmodified shards are hardlinked when permitted; EPERM/EXDEV falls back to
absolute symlinks, recorded in the receipt. Never modify shared originals.
When using Docker with symlink fallback, also bind the original checkpoint
read-only at the same absolute path inside the container. Run conversion
independently on each rank, then verify both overlays before model loading.
Keep sufficient idle GPU memory; the serving loader's guards remain mandatory.
