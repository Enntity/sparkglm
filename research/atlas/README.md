# Atlas GLM: active experimental engine

Atlas is SparkGLM's separate Rust/CUDA engine project. It is actively developed
here on `main`, under **AGPL-3.0-only**. The root installer and serving defaults
remain the **vLLM** project; nothing in this directory is enabled by default.

The [installable candidate](install/README.md) adds a pinned source build and
LLooM recipe, constrained generation, tool calls, and GLM image/video support.
Its reconstructed source is bound by `nvidia-installable-source.json`. Hardware
qualification is in progress; this is not a promoted default or a G5 release.

The native dependencies now build from pinned FlashKDA and FlashInfer source.
Focused operator parity results are recorded in the
[source-build result](../../results/candidates/2026-09-24-atlas-native-source/RESULT.md).
These checks cover operators, not full-model speed or semantic quality.

The [earlier source project](project/README.md), [September 11 report](DAY2_PROGRESS.md)
and [historical next steps](NEXT_STEPS.md) retain the prior campaign. Its best
staggered C4 result was 137.129 seconds versus its 90.500-second vLLM reference;
those older measurements do not describe the installable candidate. The older
profile depended on a cached NVIDIA object with unresolved build provenance.

The [initial bring-up](NVIDIA_REFRESH.md) and
[previous handoff](NVIDIA_HANDOFF_2026-09-11.md) retain the original fork import,
NVFP4/MTP repairs, earlier measurements and reconstruction history. The older
pre-refresh implementation below remains historical; use `project/` for new work.

## License

Atlas and the SparkGLM modifications to it are **AGPL-3.0-only**. See
`../../LICENSES/AGPL-3.0-only.txt`. FlashKDA itself is MIT and retains its
license. The SparkGLM bridge/archive glue is AGPL; the slot patch contains
MIT-derived context plus AGPL modifications. See `../../docs/LICENSING.md` for
the exact per-path boundary.

## Historical source reconstruction

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
Publication privacy replaces archived lab network defaults with required
environment variables. Supply the SSH and fabric settings for your own nodes;
this archive is not a byte-identical copy of the original private appliance
configuration. The serving algorithms are unchanged by that sanitization.

The copied `docs/` and `bench/` directories are provided for convenient web
browsing. They are also present in the reconstructable patch.
