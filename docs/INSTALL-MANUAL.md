# Install by hand

These are the same steps [`./start.sh`](../start.sh) runs, one command at a
time. Run steps 1–4 on **both** Sparks with the same `MODEL_ROOT`.

| Component | Pin |
|---|---|
| Engine | [`Enntity/atlas`](https://github.com/Enntity/atlas) `sparkglm/atlas-20261005-cvplace` @ `72588541` ([`install/atlas-source.json`](../install/atlas-source.json)) |
| Model | [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) @ `423acf37583782c51c142d145aef733d72943d93` (MIT) |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `7d74cdd881ed7e32c31175984a67823127b66cfe` (CC BY-NC-ND 4.0) |
| Native dependencies | FlashInfer `8eccd0c1`, CUTLASS `cf064d2e`, FlashKDA ([`install/flash_kda/`](../install/flash_kda/)), Rust 1.93.1, CUDA 13.0 |
| Profiles | [`install/profiles/`](../install/profiles/): `4x512k` (default) and `8x128k` |

## 1. Get the recipe

```sh
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
```

## 2. Get the image

```sh
IMAGE=$(install/build.sh --tag)
docker pull "$IMAGE" || IMAGE=$(install/build.sh)
```

The tag is the git tree hash of `install/`. `install/build.sh` needs a clean
`install/` directory. It fetches `Enntity/atlas` at the pinned commit, refuses
any other tree, and builds:

- the native bridges (FlashKDA, and NVIDIA's sparse-MLA prefill, BSD-3-Clause,
  from FlashInfer source);
- the engine for `glm-5.3-flash`;
- the engine's build-time regression tests.

A cold build downloads the CUDA base images and compiles everything; allow 30
to 60 minutes, of which the engine's kernels take about 18. With a warm
BuildKit cache it took about 5 minutes on our Sparks. `ATLAS_BUILD_JOBS`
(default 4) trades speed for memory. Keep models off the Spark while it
builds.

## 3. Download the pinned checkpoints

```sh
export MODEL_ROOT=$HOME/models/sparkglm     # the same absolute path on both Sparks
hf download nvidia/GLM-5.3-Flash-NVFP4 --revision 423acf37583782c51c142d145aef733d72943d93 \
  --local-dir "$MODEL_ROOT/nvidia--GLM-5.3-Flash-NVFP4"
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe \
  --local-dir "$MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2"
```

Instead of downloading twice, you can copy both directories to the other
Spark with `rsync -a` over the direct cable.

## 4. Convert the checkpoint overlay

Run this once per Spark. It takes about a minute and needs the GPU.

```sh
install/convert.sh "$MODEL_ROOT" "$IMAGE"
```

Atlas serves the NVIDIA checkpoint through `$MODEL_ROOT/atlas-overlay`. The
overlay quantizes the 864 MTP matrices Atlas needs to NVFP4 and links
everything else back to the original files. The script verifies every payload
and records the converter it used. It skips the work when the overlay already
matches that converter; otherwise it moves any older overlay aside.

## 5. Start the pair, worker first

Start rank **1** (the worker) first, then rank **0** (the leader). The
`--leader-address` is rank 0's IPv4 address on the direct cable. Each rank
finds its own fabric interface from the RoCE device (`--fabric-hca`, default
`rocep1s0f0`). Pass `--fabric-interface` to override it.

```sh
# on rank 1:
install/start-node.sh --rank 1 --leader-address 192.0.2.1 --model-root "$MODEL_ROOT" --image "$IMAGE"
# then on rank 0:
install/start-node.sh --rank 0 --leader-address 192.0.2.1 --model-root "$MODEL_ROOT" --image "$IMAGE"
until curl -sf http://127.0.0.1:8893/health; do sleep 10; done
```

**Loading and warmup.** Loading takes about 2.5 minutes. The first start fills
the CUDA kernel cache (`~/.cache/atlas-cuda`), and so do the first requests of
each shape; later starts reuse it. Run one warmup pass before measuring.

**Profiles.** `--profile 8x128k` selects the other shipped profile. A path to a
JSON file runs your own profile. Use the same profile on both ranks.

**Memory share.** `--gpu-memory-utilization 0.91` on both ranks gives a KV
pool of about 1.13M tokens on Sparks that run nothing else; without it the
profile's 0.88 gives about 600K. We run 0.91: above it, real traffic took our
rank 0 below 1 GiB free.

**Prefix cache on disk.** `--prefix-cache-dir DIR` (and optionally
`--prefix-cache-gb 48`) on both ranks keeps evicted prefix-cache entries on
each node's disk; see the README for what it costs.

**Display memory as KV cache.** `--display-carveout` on both ranks lends each
Spark's 2 GiB display carveout to the KV cache (about 1.37M tokens at 0.91).
The container starts with `CAP_SYS_ADMIN` to export it, which the server
drops before it loads anything, and shares the lock directory
`/run/lock/sparkglm`. It needs a validated driver (580.173.02 or 580.178.04)
and a headless Spark; see the README.

**Stopping.** `docker rm -f atlas-sparkglm-rank0` (and `-rank1`).

## Troubleshooting

- **NCCL fails with `modify_qp -> RTR failed`:** the RoCE link, or its IPv4
  address, is missing on one Spark. The engine picks each rail's RoCE v2 IPv4
  GID automatically. To force one, add `"ATLAS_RDMA_GID": "<index>"` to the
  profile's `environment`; the launcher ignores ambient `ATLAS_*` variables.
- **The host stops responding under load:** GB10 memory is shared with the
  host. Keep builds, other models and large processes off the pair while
  serving. The default profile leaves about 8–9 GB free at full KV.
- **A rank exits during loading:** `docker logs atlas-sparkglm-rank0`, or
  `./start.sh logs worker`. The launcher refuses to start without a verified
  overlay and a complete drafter.
