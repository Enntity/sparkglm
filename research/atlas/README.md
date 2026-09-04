# Atlas-native GLM archive

This is the preserved first-principles Rust/CUDA implementation built on Atlas.
It contains valuable GLM contracts, typed-state work, native probes, KDA/DSA
experiments, and lessons that informed the current vLLM line. It did not reach
competitive end-to-end serving performance and is not the recommended engine.

## License

Atlas and the SparkGLM modifications to it are **AGPL-3.0-only**. See
`../../LICENSES/AGPL-3.0-only.txt`. FlashKDA material under `flash_kda/` is MIT
and retains its own license.

## Exact reconstruction

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Atlas-Inf/atlas.git atlas-sparkglm
cd atlas-sparkglm
git checkout bdcccc2ca91eba084aac94a059e3b0f4a5d556dd
git apply /path/to/sparkglm/research/atlas/atlas-glm53.patch
```

The patch represents SparkGLM Atlas archive commit
`775cb3655e29a3735f4f58faa540608f9427bf51`, excluding the prebuilt
`libatlas_glm53_flash_kda.so`. Rebuild that artifact from the pinned source
using `flash_kda/rebuild.sh`; compiled binaries are intentionally not
distributed in this repository.

The copied `docs/` and `bench/` directories are provided for convenient web
browsing. They are also present in the reconstructable patch.
