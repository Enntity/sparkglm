# Atlas GLM: active experimental engine

Atlas is SparkGLM's separate Rust/CUDA engine project. It is actively developed
here on `main`, under **AGPL-3.0-only**. The root installer and serving defaults
remain the **vLLM** project; nothing in this directory is enabled by default.

The current NVIDIA NVFP4 campaign has working native MTP2, 32K context and four
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
