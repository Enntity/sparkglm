# Install by hand

These are the same steps [`./start.sh`](../start.sh) runs, one command at a
time. Run steps 1–3 on **every** participating Spark with the same `MODEL_ROOT`.
The unified TP2/TP4 image is an unpublished RC; final-image TP4 qualification
is pending. The launcher instructions describe supported configuration, not
completed hardware validation.

| Component | Pin |
|---|---|
| Engine | [`Enntity/atlas`](https://github.com/Enntity/atlas) exact commit and tree in [`install/atlas-source.json`](../install/atlas-source.json) |
| Model | [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) @ `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` (165.3 GiB, NVIDIA Open Model Agreement; base model MIT) |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `7d74cdd881ed7e32c31175984a67823127b66cfe` (CC BY-NC-ND 4.0) |
| Native dependencies | FlashInfer `8eccd0c1`, CUTLASS `cf064d2e`, FlashKDA ([`install/flash_kda/`](../install/flash_kda/)), Rust 1.93.1, CUDA 13.0 |
| Profiles | [`install/profiles/`](../install/profiles/): `4x1m` (what `./start.sh` runs), `4x512k` (the `start-node.sh` default), `8x128k` and TP4 `4x1m-tp4` |

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
BuildKit cache it takes about 3 minutes on our Sparks. `ATLAS_BUILD_JOBS`
(default: one job per ~10 GiB of free memory, at most half the cores) trades
speed for memory, and `ATLAS_IMAGE_CHECKS=0` skips the in-image release checks
for an engine commit that already passed them. Keep models off the Spark while
it builds.

## 3. Download the pinned checkpoints

```sh
export MODEL_ROOT=$HOME/models/sparkglm     # the same absolute path on every Spark
hf download nvidia/GLM-5.3-Flash-FP4-Local --revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 \
  --local-dir "$MODEL_ROOT/nvidia--GLM-5.3-Flash-FP4-Local"
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe \
  --local-dir "$MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2"
```

Instead of downloading on every node, copy both directories to the other
Sparks with `rsync -a` over the fabric.

FP4-Local needs no on-disk overlay. Strict preserves its weight formats;
Spark makes its selected conversions in memory when loading.

**The earlier NVFP4 checkpoint** ([`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
@ `423acf37583782c51c142d145aef733d72943d93`, MIT, 190.4 GiB) remains an
installer override outside the unified Strict/Spark RC qualification;
required native-format checks still apply at load. To prepare it, download
to `$MODEL_ROOT/nvidia--GLM-5.3-Flash-NVFP4`, then
convert its overlay once per Spark (about a minute, needs the GPU):

```sh
install/convert.sh "$MODEL_ROOT" "$IMAGE"
```

Atlas serves that checkpoint through `$MODEL_ROOT/atlas-overlay`, which
quantizes the 864 BF16 MTP matrices Atlas needs to NVFP4 and links everything
else back to the original files. The script verifies every payload and records
the converter it used. It skips the work when the overlay already matches that
converter; otherwise it moves any older overlay aside. Pass
`--checkpoint nvidia--GLM-5.3-Flash-NVFP4` to `start-node.sh` on both ranks
below (`./start.sh` does both when `.env` sets `CHECKPOINT_REPO` and
`CHECKPOINT_REVISION`).

## 4. Start the pair, worker first

Start rank **1** (the worker) first, then rank **0** (the leader). The
`--leader-address` is rank 0's IPv4 address on the direct cable. Each rank
uses the explicitly selected local RDMA device (`--fabric-hca`) or derives it
from an unambiguous `--fabric-interface`. No physical port number is assumed.
A comma-separated `--fabric-hca` list uses its first device for the socket
interface and passes the full list to the engine. The list is exact: other
PCIe-domain devices are not added automatically.

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

**Profiles.** `start-node.sh` defaults to `4x512k`. `--profile 4x1m` runs what
`./start.sh` runs (four requests of up to 1,048,576 tokens), and
`--profile 8x128k` selects the third shipped profile. A path to a JSON file
runs your own profile. Use the same profile on every rank. `--mode strict`
is the default; `--mode spark` selects the performance policy described in
the [README](../README.md#strict-and-spark-modes). Native MTP is opt-in with
`--mode strict --strict-mtp`; it supports C1..C4 and retains prefix caching.
The eight-slot `8x128k` profile is not a native-MTP configuration.

**Memory share.** `--gpu-memory-utilization 0.91` on both ranks gives the KV
pool much more room on Sparks that run nothing else (with the NVFP4 checkpoint,
about 1.13M tokens unsplit against 600K at the profile's 0.88). We run 0.91:
above it, real traffic took our rank 0 below 1 GiB free.

**KV shard.** `--kv-shard` on both ranks splits the KV cache between the two
Sparks: each stores the latents of half the blocks, so the pool holds about
1.8x the tokens, and the output is bit-identical to the unsplit pool's.
`./start.sh` turns it on by default; `start-node.sh` leaves it off unless you
pass the flag. At 0.91 with the display carveout and the disk prefix cache the
pool is about 2.66M tokens with FP4-Local (2.34M with the NVFP4 checkpoint).

**Prefix cache on disk.** `--prefix-cache-dir DIR` (and optionally
`--prefix-cache-gb 48`) on both ranks keeps evicted prefix-cache entries on
each node's disk; see the README for what it costs.

**Display memory as KV cache.** `--display-carveout` on both ranks lends each
Spark's 2 GiB display carveout to the KV cache (about 28% more blocks with
`--kv-shard` at 0.91).
The containers share the lock directory `/run/lock/sparkglm`. It needs an
NVIDIA driver that lends the display memory (580.76.05 or later) and a
headless Spark; see the README.

**Stopping.** `docker rm -f atlas-sparkglm-rank0` (and `-rank1`).

## Four-Spark chain or ring

Use the same image, model directories and `4x1m-tp4` profile on all four nodes.
`start.sh` remains the two-node convenience launcher; TP4 uses one
`install/start-node.sh` invocation per rank, directly or through an orchestrator.
Start workers 1, 2 and 3 before leader 0. All ranks need the same reachable leader
IPv4 address and a selected socket network interface.

`--world-size 4` defaults to `--topology chain`. A ring requires
`--topology ring` on every rank and physical links closing rank 3 to rank 0.
Left and right refer to rank order:

| Rank | Chain left / right | Ring left / right |
|---|---|---|
| 0 | none / 1 | 3 / 1 |
| 1 | 0 / 2 | 0 / 2 |
| 2 | 1 / 3 | 1 / 3 |
| 3 | 2 / none | 2 / 0 |

Pass an exact comma-separated HCA list for each connected side with
`--left-hca` and `--right-hca`. On a dual-rail fabric, list both rails in
matching peer order on each edge, including both rails of the ring's closing
3–0 edge. Left and right must use distinct devices on a node. HCA names may
differ between nodes; choose them from the actual wiring and RoCE addresses.
The launcher does not configure cables, addresses or host routes.

For example, on each ring node, after assigning its local values:

```sh
install/start-node.sh --world-size 4 --topology ring --rank "$RANK" \
  --leader-address "$LEADER_ADDRESS" --fabric-interface "$SOCKET_INTERFACE" \
  --left-hca "$LEFT_RAILS" --right-hca "$RIGHT_RAILS" \
  --model-root "$MODEL_ROOT" --image "$IMAGE" --profile 4x1m-tp4 --mode spark
```

For a chain, use `--topology chain`, omit `--left-hca` on rank 0 and omit
`--right-hca` on rank 3. To select native MTP instead, replace `--mode spark`
with `--mode strict --strict-mtp` on every rank. `--fabric-hca` belongs to the
TP2 launcher; four ranks use the explicit left/right lists. Missing required
links, unexpected endpoint links or overlapping local device lists are refused.
A failed ring setup does not silently become a chain.

Managed containers express the same settings as `NODE_RANK`,
`SPARKGLM_TOPOLOGY=chain|ring`, `FABRIC_LEFT_HCA`, `FABRIC_RIGHT_HCA`,
`FABRIC_INTERFACE`, `SPARKGLM_MODE`, and optional `SPARKGLM_STRICT_MTP=1`.
The profile supplies world/TP/EP4. See [LLooM integration](LLOOM.md).

The profile keeps four request slots, a 1,048,576-token per-request limit and
memory utilization 0.72. Capacity depends on actual free memory and the chosen
mode. Strict native MTP also allocates a private predictor cache on rank 0;
it does not inherit the TP2-specific 66,000-block cap. Read startup allocation
and load-test receipts before changing the memory setting.

When disk prefix caching is enabled, TP2 uses `strict/` or `spark/` beneath
the mounted cache root. TP4 uses `tp4/chain/strict/`, `tp4/chain/spark/`,
`tp4/ring/strict/` or `tp4/ring/spark/`; each contains the separate KV/state
directories. This prevents cache reuse across modes, rank counts or topologies.
The engine currently refuses a TP4 token-sharded KV pool together with the
NVMe KV tier; use disk caching without `--kv-shard` on TP4.

## Troubleshooting

- **NCCL fails with `modify_qp -> RTR failed`:** the RoCE link, or its IPv4
  address, is missing on one Spark. The engine picks each rail's RoCE v2 IPv4
  GID automatically. To force one, add `"ATLAS_RDMA_GID": "<index>"` to the
  profile's `environment`; the launcher ignores ambient `ATLAS_*` variables.
- **`no RoCE v2 GID with an IPv4 address` or `ibv_modify_qp failed with 22`
  on `roceP2p1s0f0`:** the board wires the ConnectX-7's second PCIe domain to a
  separate physical port rather than to the same cable (ASUS GX10). An
  unaddressed second port is now left out automatically; if it has an address
  but no cable to the other Spark, use an exact `FABRIC_HCA` list in `.env` (or pass
  `--no-fabric-siblings` to `install/start-node.sh` on both ranks). The fabric
  then runs on the one cabled device: one RDMA rail instead of two.
- **The host stops responding under load:** GB10 memory is shared with the
  host. Keep builds, other models and large processes off the pair while
  serving. The default profile leaves about 8–9 GB free at full KV.
- **A rank exits during loading:** `docker logs atlas-sparkglm-rank0`, or
  `./start.sh logs worker`. The launcher refuses to start without a complete
  checkpoint and drafter, and refuses the NVFP4 checkpoint without its
  verified overlay.

## TP2 per-node fabric selection

`./start.sh fabric` runs a read-only preflight on both hosts. It reports their
selected HCA lists, network interfaces and IPv4 addresses. `./start.sh` performs
the same check before preparation or stopping existing containers. A missing,
down or mismatched override fails. Zero matching pairs, multiple matches
(including multiple addresses on one interface), or ambiguous extra rails also
fail with an error. Active backchannels are not presumed to reach the peer.

Set `LEADER_FABRIC_HCA`, `WORKER_FABRIC_HCA`, `LEADER_FABRIC_INTERFACE` and
`WORKER_FABRIC_INTERFACE` independently in `.env`. A side without its own setting
inherits the corresponding legacy `FABRIC_HCA` or `FABRIC_INTERFACE` value.
If an HCA list is supplied, its first device selects the socket interface;
list every desired rail in matching peer order on both nodes. For example,
one host can use `rocep1s0f1,roceP2p1s0f1` and the other
`rocep1s0f0,roceP2p1s0f0`. Each listed pair must have a unique matching IPv4 subnet.
The same physical port number or HCA name on both machines is not required.
`LEADER_ADDRESS`, if supplied, must match the selected leader interface.

The launch scripts pass exact lists with `FABRIC_SIBLINGS=0`. In particular,
select both intended rails explicitly on dual-domain systems; automatic sibling
expansion could select a backchannel on other board layouts. Direct TP2 container
launches must supply `FABRIC_HCA`; a missing value fails instead of using port 0.
