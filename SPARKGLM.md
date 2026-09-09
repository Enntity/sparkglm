# Run SparkGLM independently on two DGX Sparks

SparkGLM installs and serves directly with **Docker and SSH**. It does not
install or require LLooM. NVFP4 is the default: 524288-token context, 9 GiB KV
per rank, native FlashInfer CUTLASS, MXFP8 DFlash2 TP2 k7, 2K chunks and mixed
scheduling. `--skip` selects the recorded NVFP4 comparison option;
`--profile exl3` selects the latest concurrent E3 EXL3 profile with 1M context.

This is a source research preview, not a G5-certified appliance. Read
[the results](results/CURRENT.md) and [model licenses](docs/LICENSING.md).

## Prerequisites

On both Linux ARM64 DGX Sparks: Docker with NVIDIA Container Toolkit, Python 3
with venv, Git, rsync, and passwordless SSH from leader to worker. Use matching
account home paths and the same absolute model-root path. Configure the direct
TP fabric and inspect `ip address`, `ibdev2netdev` and the populated RoCE v2 GID
tables to obtain each node's interface, RDMA device and GID index.

Stop other resident full models through their current manager before building
or installing. Native compilation requires at least 32 GiB MemAvailable on the
leader. Allow roughly 200 GB for the target plus the draft, source/build layers,
images and caches on each node; 190 GB total free space is not sufficient.
The API binds to the supplied leader fabric address; keep it on your trusted
private network or put an authenticated gateway in front of it.

## Install

Run on the leader. Replace the uppercase placeholders with your configuration:

```bash
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
./start.sh plan
./start.sh --worker USER@WORKER \
  --head-address HEAD_FABRIC_IP --worker-address WORKER_FABRIC_IP \
  --interface HEAD_INTERFACE --worker-interface WORKER_INTERFACE \
  --hca HEAD_RDMA_DEVICE --worker-hca WORKER_RDMA_DEVICE \
  --gid HEAD_GID_INDEX --worker-gid WORKER_GID_INDEX
```

The worker interface, HCA and GID default to the leader's values when their
worker overrides are omitted. The installer builds the pinned source layers,
copies the immutable image, downloads pinned weights into
`~/.cache/sparkglm/models`, copies the weights locally to the worker, and starts
worker then leader. `--model-root /path/to/models` reuses your existing weight
location; `--image sha256:FULL_IMAGE_ID` reuses a qualified local build.
No daemon or package manager beyond Docker is required for serving.

The default endpoint is `http://HEAD_FABRIC_IP:8890/v1`, model
`sparkglm-nvfp4`. A successful start checks the exact model identity. Its
standalone containers are named `sparkglm-standalone-nvfp4-head` and
`sparkglm-standalone-nvfp4-worker`; use `docker logs` on their respective nodes
for startup diagnostics. The default readiness budget is two hours because a
first source build/model load is not a quick prebuilt-image installation.

```bash
./start.sh status
./start.sh stop
./start.sh start
```

Lifecycle commands read the saved local standalone installation. They operate
only on containers bearing SparkGLM's standalone ownership label. There is no
independent auto-restart policy; an unsuccessful launch stops its ranks. Keep
the same worker and profile when restarting. For EXL3, include `--profile exl3`
in each command. The old EXL3 video-foundation launcher remains separately
available as `start-exl3.sh`; see [its historical guide](docs/EXL3_QUICKSTART.md).

Build pins are in [profiles/build.json](profiles/build.json) and
[profiles/build-exl3.json](profiles/build-exl3.json). The
[public source map](provenance/2026-09-08-publication-map.json) connects them to
the measured original source trees. Rebuilding does not promise bit-identical
images; hardware qualification remains necessary for a new build.

## Optional LLooM integration

If you already use LLooM, it can own the same SparkGLM runtime instead:

```bash
./start.sh --lloom --worker USER@WORKER
./start.sh --lloom status
```

See [LLooM integration](docs/LLOOM_INSTALL.md) for its cluster prerequisites and
immutable-image recipe. SparkGLM does not install LLooM even in this mode.
Choose one lifecycle owner; stop the active standalone or managed runtime
before switching. This changes management, not the measured inference recipe.

## Experimental NVIDIA checkpoint

The optional `--profile nvfp4-nvidia` selects the separately pinned NVIDIA
ModelOpt checkpoint and its own runtime ID. It does not replace the Red Hat
default. Read the [experiment and qualification status](research/experiments/nvfp4-nvidia/README.md)
before installation.
