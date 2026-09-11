# Portable r11b resume assets

This disabled research profile preserves the final r11b runtime flags. It
renders Docker argv only: it never starts a container, contacts a host, drains
requests or changes another service. The two-rank profile reserves four 32K
slots with repaired MTP2, BF16 KV/head, FP32 SSM, 4096-token prefill chunks,
GPU utilization 0.914, a 4096 MiB free-memory guard and 114 GiB container
memory/swap limits. `ATLAS_FP4_PREFILL` is absent, not set to zero.

The completed r11b load had 11,189 target blocks on rank 0 and 10,336 on rank
1, above the 8,196 floor; each rank had 8,196 private speculative blocks.
The earlier 0.91 attempt failed the target floor. These are observations on
the measured appliance, not guaranteed free capacity on another installation.
The floor and post-allocation checks must remain enabled.

`profile.json` preserves all runtime flags, including apparently redundant
upstream options. Do not prune or independently enable flags named `UNSAFE`:
they belong to the pinned repaired GLM policy, and this profile is not a
general safe configuration for other models. The standalone renderer changes
only rank/site locators, image identity, container name and filesystem mounts;
the private Nsight wrapper and its mounts are omitted. All inference settings
are compared against the exact verified manifest in the static test.

## Reconstruct and prepare the external overlay

First reconstruct the engine using [the handoff](../NVIDIA_HANDOFF_2026-09-11.md).
In the commands below, set `SPARKGLM_SOURCE` to the absolute public checkout
path, `ATLAS_SOURCE` to the reconstructed engine, `CHECKPOINT` to the pinned
local checkpoint and `OVERLAY` to a separate output directory. Set these on
each host; the renderer does not discover or create directories.
The final source checkpoint is `dd0ffd157fb7c6e96d2f3858c888513c6f142cf6`,
tree `e6a4b26058000d1c81b0c87bbf4c2cb0c9127c03`. It is exported as patches;
it is not a commit to fetch from Mango's upstream remote.

Follow the [MTP converter instructions](../nvidia-mtp-converter/README.md) to
build its native helper, run its one-matrix check, then produce and verify the
overlay from NVIDIA checkpoint revision
`423acf37583782c51c142d145aef733d72943d93`:

```sh
python3 "$SPARKGLM_SOURCE/research/atlas/nvidia-mtp-converter/convert.py" \
  --source "$CHECKPOINT" --output "$OVERLAY" --one-matrix
python3 "$SPARKGLM_SOURCE/research/atlas/nvidia-mtp-converter/convert.py" \
  --source "$CHECKPOINT" --output "$OVERLAY"
python3 "$SPARKGLM_SOURCE/research/atlas/nvidia-mtp-converter/convert.py" \
  --source "$CHECKPOINT" --output "$OVERLAY" --verify-overlay
```

The conversion is a separate, deliberate GPU operation. Never run it alongside
a loaded model. Model weights and generated tensors stay outside Git. On each
host, the original checkpoint is mounted read-only at its same absolute path,
so overlay symlinks resolve inside the container. Verify the overlay on each
host after copying it; do not assume absolute links survive relocation.

Use the public `files/chat_template.jinja` as the `--template` argument. Its
SHA-256 is `2e24c13c22cbd63370908e2302a045bceeac801ff83d28ec683d1983175a3886`,
identical to the tested template. It mounts over both model and engine template
locations. The template retains its separate GLM license/provenance.

## Build, then run focused native tests

`Dockerfile` adapts the pinned Mango model Dockerfile, with base-image digests
from its GB10 Dockerfile. Rust 1.93.1, CUDA 13.0, NCCL
`2.31.2-1+cuda13.3`, and CUTLASS
`cf064d2e6bad2886238ac565b3b49007764f4939` are pinned. Build on native aarch64
DGX Spark using the reconstructed engine as context; inherited build scripts
select the GB10 CUDA architecture. No compiled artifacts are supplied.

**This exact clean-image recipe is source-reviewed, not freshly built and
tested.** The measured r11b binary came from the cached native build. Public
base/package availability and a clean image build still need verification;
do not label a newly built image as the measured binary without its own tests.

```sh
docker build --target builder \
  -f "$SPARKGLM_SOURCE/research/atlas/nvidia-prefill/Dockerfile" \
  -t atlas-nvidia-r11b-builder "$ATLAS_SOURCE"
docker run --rm --entrypoint cargo atlas-nvidia-r11b-builder \
  test --locked --release -p spark-server --bin spark sanitizer
docker run --rm --entrypoint cargo atlas-nvidia-r11b-builder \
  test --locked --release -p spark-server --bin spark reasoning_done
docker build \
  -f "$SPARKGLM_SOURCE/research/atlas/nvidia-prefill/Dockerfile" \
  -t atlas-nvidia-r11b "$ATLAS_SOURCE"
```

The recorded native filters passed 42 sanitizer tests and four reasoning-Done
tests. These CPU-path tests link the native libraries but do not validate GPU
arithmetic. Run the relevant native GPU/index tests separately before a new
model qualification; do not substitute a stub-kernel build for those gates.

## Render worker first, then leader; execute manually

Prepare the checkpoint, overlay, template and writable CUDA-cache directory
on each host. Replace image tag with the verified local image identity. Set
the fabric address, interface and HCA for that host explicitly. No service
management, remote execution or inference is performed by these commands:

```sh
python3 "$SPARKGLM_SOURCE/research/atlas/nvidia-prefill/render.py" \
  --rank 1 --master "${MASTER_ADDR:?}" --bind 127.0.0.1 --port 8894 \
  --nic "${FABRIC_NIC:?}" --hca "${RDMA_HCA:?}" \
  --name atlas-nvidia-r11b-worker --image "${ATLAS_IMAGE:?}" \
  --overlay "${OVERLAY:?}" --original "${CHECKPOINT:?}" \
  --cuda-cache "${CUDA_CACHE_DIR:?}" --template "${CHAT_TEMPLATE:?}" \
  --format shell > worker-command.txt
python3 "$SPARKGLM_SOURCE/research/atlas/nvidia-prefill/render.py" \
  --rank 0 --master "${MASTER_ADDR:?}" --bind 127.0.0.1 --port 8893 \
  --nic "${FABRIC_NIC:?}" --hca "${RDMA_HCA:?}" \
  --name atlas-nvidia-r11b-leader --image "${ATLAS_IMAGE:?}" \
  --overlay "${OVERLAY:?}" --original "${CHECKPOINT:?}" \
  --cuda-cache "${CUDA_CACHE_DIR:?}" --template "${CHAT_TEMPLATE:?}" \
  --format shell > leader-command.txt
```

Inspect both commands, then run the worker command on its host and the leader
command on its host, only after the appliance is explicitly available. Do not
pipe renderer output directly into a shell. Check both ranks' readiness and
actual KV allocations before a short canary. Follow with the frozen field-guide
request and preserve SSE timing, usage, content and reasoning separately.

## Static verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  "$SPARKGLM_SOURCE/research/atlas/nvidia-prefill/test_profile.py"
```

The author also ran this test with `--verified-manifest` pointing to the private
original receipt. Its SHA-256 is stored in `profile.json`; the test checks
every environment key and every server argument after replacing only site
network locators, and checks fixed Docker isolation/memory options. Names,
image/path mappings and optional Nsight instrumentation are outside runtime
flag parity. The private receipt is deliberately not distributed. The public
test still checks both ranks, critical limits and absence of FP4 prefill.
