# Atlas GLM: active experimental engine

Atlas is SparkGLM's separate Rust/CUDA engine project. It is actively developed
here on `main`, under **AGPL-3.0-only**. The root installer and serving defaults
remain the **vLLM** project; nothing in this directory is enabled by default.

## Current installable build (2026-09-29)

The current Atlas candidate is built from the public
[`Enntity/atlas`](https://github.com/Enntity/atlas) fork
(`sparkglm/atlas-20260928`: Atlas-Inf `main` plus our GLM-5.3 Flash layer and
one SparkGLM-only bridge commit) and installed with the recipe on branch
[`atlas/installable-20260929`](https://github.com/Enntity/sparkglm/tree/atlas/installable-20260929). Follow its
[install and reproduction guide](https://github.com/Enntity/sparkglm/blob/atlas/installable-20260929/research/atlas/install/README.md): it
builds every component from pinned sources, starts the two-Spark pair with or
without LLooM, and reproduces the measurements in the
[result bundle](https://github.com/Enntity/sparkglm/tree/atlas/installable-20260929/results/candidates/2026-09-29-atlas-merged).

Measured with that recipe on our pair (candidate evidence, not a
qualification): the frozen matrix summed to 154.8 s (mean of 3; vLLM
SparkGLM 206.6 s) and the staggered C4 field guide took 57.5 s (median of 3;
vLLM SparkGLM adaptive 70.5 s). vLLM remains the recommended default.

## Earlier campaign (2026-09-11, historical)

The NVIDIA NVFP4 campaign below had working native MTP2, 32K context and four
owners. The best retained staggered C4 took **137.129 seconds**, versus
**205.468 seconds** for the first full run and **90.500 seconds** for the vLLM
reference. These are bounded research measurements, not complete qualification
or a claim of parity. See the [checksum-bound result](../../results/candidates/2026-09-11-atlas-day2/RESULT.md).

Start with the [current source and operator project](project/README.md),
[day's chronological report](DAY2_PROGRESS.md), and [next steps](NEXT_STEPS.md).
The latest source includes a diagnostic after the best measured candidate;
source, runtime profiles and diagnostic/performance results remain distinguished.
The fastest measured native-prefill configuration depends on an excluded cached
NVIDIA object whose exact source/build identity is unresolved. The bridge source
is preserved, but that configuration is not a self-contained public binary build.

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
