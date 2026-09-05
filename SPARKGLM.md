# Build and run SparkGLM

The root recipe is the current two-GB10 vLLM/EXL3 engine. Atlas and alternative
build experiments under `research/` are not required to run it.

## Before downloading

Read [the licensing boundaries](docs/LICENSING.md). The default matches the
published video: DFlash2 k=7, grouped prefill, cooperative decode, and mixed
scheduling. The separately fetched DFlash2 checkpoint is **CC BY-NC-ND 4.0**
(non-commercial/no-derivatives). Choose `SPEC_METHOD=mtp` or `none` if those
terms do not fit; those profiles do not inherit the video's performance.
The target model and quantization have their own license obligations.

This is a research preview, not a fully qualified production appliance. Read
[known limitations](docs/KNOWN_LIMITATIONS.md), especially the sparse-MLA
candidate-set approximation, before deploying it.

## Prepare the two Sparks

Use two DGX Spark GB10 systems with a working CX7/RoCE connection, Docker with
NVIDIA GPU support on both, and key-based SSH from head to worker. The head
also needs Python 3, the Hugging Face CLI, and rsync. Allow disk space on both
nodes for the approximately 164 GiB target, drafter, image, and build caches.
See the [upstream setup guide](docs/upstream/MIA_RECIPE_README.md) for host
prerequisites; use this repository's defaults and image for SparkGLM.

On the head:

```bash
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
cp .env.example .env
```

Edit `.env` before launching: set `HEAD_IP`, `WORKER_IP`, any worker-account
overrides, and each node's CX7 interface, HCA, and GID index. Example interface
names are not universal. The launcher checks each node's networking.
Keep `.env` private; never commit credentials or machine-specific settings.

Stop other resident model servers before building or serving. Native
compilation shares the GB10's unified memory with model weights; the build
refuses less than 32 GiB available RAM by default. Do not disable that guard
just to compile alongside a loaded model.

## Build and launch

```bash
BUILD=1 ./start.sh
```

This builds `sparkglm:local` from the root Dockerfile, ships the same image to
the worker, downloads the pinned weights, syncs them, launches TP2, and runs
startup warmup. The first source build/download can be substantial; this is
not a seconds-long installation. Subsequent `./start.sh` runs reuse matching
images, weights, and persistent JIT caches. Don't set `SKIP_BUILD=1` when
trying to reproduce this checkout's implementation.

For an alternative to the DFlash2 default, select it **before the first run**:

```bash
SPEC_METHOD=mtp BUILD=1 ./start.sh
# Or disable speculation: SPEC_METHOD=none BUILD=1 ./start.sh
```

The API binds to `127.0.0.1:8888` by default. Check it from the head:

```bash
./start.sh status
curl --fail http://127.0.0.1:8888/health
./start.sh logs
```

Use `./start.sh logs worker` for rank 1, `./start.sh stop` to stop both recipe
containers, and `./start.sh restart` after configuration changes. A custom
`PORT` changes the example URL. To expose the API beyond loopback, configure
`API_HOST` **and** `VLLM_API_KEY`; also protect the host-networked TP/RoCE
fabric from untrusted clients. Do not use the unauthenticated-LAN escape hatch
on shared networks.

Existing `.env` files are preserved, not upgraded automatically. Compare yours
with `.env.example` when changing revisions.

## What the build includes

The source-built image includes the native FP16 sparse-selector/top-k and
DeepGEMM support, M64 fat-expert path, grouped prefill, cooperative decode,
and the retained scheduler/cache changes. These are not Python-only overlays.
The build checks the video engine's source manifest; current source labels,
chat-template integration, and launcher safety work remain explicit changes.
See [the video profile](docs/PUBLISHED_VIDEO_CONFIGURATION.md) for exact pins
and [the source audit](docs/VIDEO_SOURCE_RESTORATION.md) for its boundary.

Credit and source revisions are in [provenance](docs/PROVENANCE.md) and
[attribution](docs/ATTRIBUTION.md). Logical development commits remain in
`research/current-engine-history/patches/`; the publication branch has a
sanitized history without upstream model artifacts.

## Test before changing defaults

Run `./scripts/check.sh all` for source checks. Follow
[the methodology](docs/METHODOLOGY.md) for exact-shape tests, tinyGLM, then
matched full-model and quality gates. The [current evidence status](results/CURRENT.md)
distinguishes retained measurements from complete release qualification.
