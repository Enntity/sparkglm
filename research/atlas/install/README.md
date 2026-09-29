# Install and reproduce Atlas SparkGLM

GLM-5.3-Flash (NVIDIA NVFP4) on two directly connected NVIDIA DGX Spark (GB10)
systems, served by the Atlas engine with tensor/expert parallelism across both
boxes and DFlash2 speculative decoding. Everything is built from pinned public
sources; no binaries or weights are distributed here.

| Component | Pin |
|---|---|
| Engine | [`Enntity/atlas`](https://github.com/Enntity/atlas) `sparkglm/atlas-20260928` @ `c9723935592f3d12215a77b0dfdfcadf641a567f` (tree `4cd94911`) — see [`../atlas-source.json`](../atlas-source.json) |
| Engine layers | Atlas-Inf `main` @ `6a3d24ec` + our upstream candidate `upstream/glm53-flash` @ `5c75696b` + one SparkGLM-only commit (FlashKDA and native sparse-MLA prefill bridges) |
| Model | [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) @ `423acf37583782c51c142d145aef733d72943d93` |
| Drafter | [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `7d74cdd881ed7e32c31175984a67823127b66cfe` |
| Native deps | FlashInfer `8eccd0c1`, CUTLASS `cf064d2e`, FlashKDA (see `../flash_kda/rebuild.sh`), Rust 1.93.1, CUDA 13.0 |
| Launch profile | [`profile.json`](profile.json): 4 concurrent requests × 512K context, FP8-latent KV, DFlash2 γ=8, KDA record rollback, GPU utilization 0.88 |

## What we measured with this recipe

Two DGX Sparks, one 200G ConnectX-7 cable, 2026-09-29. Full receipts and
caveats: [`results/candidates/2026-09-29-atlas-merged/`](../../../results/candidates/2026-09-29-atlas-merged/RESULT.md).

| Workload | This recipe | Reference |
|---|---|---|
| Matrix (C1/C2 16K+32K, C4 16K; 400 tokens each), sum of walls, mean of 3 | **152.9 s** | vLLM SparkGLM 206.6 s |
| Staggered C4 field guide (4 × ~16K, arrivals 0/1/2/3 s), median of 3 | **58.0 s** | vLLM SparkGLM adaptive 70.5 s · Mia EXL3 113.2 s |
| Single-stream decode (mmastrac prompts: structured / code / prose) | 83.5 / 59.9 / 31.9 tok/s | — |
| Cold prefill (8K / 32K / 64K / 128K) | ~2,480 / 2,450 / 2,290 / 2,070 tok/s | — |
| Long context | 4 × 190K concurrent and 1 × 500K complete, ≥ 8.3 GB host memory free | — |
| Hard quality probe (arithmetic / two-hop 24K needle) | 40/40 · 11/12 | — |

These are measurements on our pair, not a qualification of your hardware.

## Requirements

- Two DGX Spark (GB10, 128 GB) systems joined by a direct ConnectX-7 cable with
  RoCE working: an IPv4 address on the fabric interface of each node
  (`ib_write_bw` between them should succeed). The engine discovers the RoCE
  v2 GID on its own and uses both PCIe halves of the port.
- Docker with the NVIDIA container runtime and BuildKit (Docker 25+), `git`,
  `python3`, and ~260 GB free disk per node (checkpoint 203 GB, overlay 16 GB,
  image ~3.5 GB, build cache).
- No other model loaded while building, converting or serving: the build and
  the engine both need most of the GB10's unified memory.

Run steps 1–4 on **both** nodes, with the same paths on each.

## 1. Get the recipe

```sh
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
git checkout atlas/installable-20260929
```

## 2. Download the pinned checkpoints

```sh
export MODEL_ROOT=/srv/models        # any absolute path, same on both nodes
pip install -U "huggingface_hub[cli]"
hf download nvidia/GLM-5.3-Flash-NVFP4 --revision 423acf37583782c51c142d145aef733d72943d93 \
  --local-dir "$MODEL_ROOT/nvidia--GLM-5.3-Flash-NVFP4"
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe \
  --local-dir "$MODEL_ROOT/incoai--GLM-5.3-Flash-DFlash2"
```

Check the licenses of both checkpoints before downloading (see
[`docs/LICENSING.md`](../../../docs/LICENSING.md)); the DFlash2 drafter is
non-commercial.

## 3. Build the image

```sh
bash research/atlas/install/build.sh "$HOME/atlas-install"
```

The script requires a clean checkout. It fetches `Enntity/atlas` at the pinned
commit, refuses any other tree, builds the native bridges (FlashKDA and the
NVIDIA sparse-MLA prefill from FlashInfer source), the engine (target
`glm-5.3-flash`) and its build-time regression tests, and prints the image
tag `lloom/atlas-sparkglm:<sparkglm-revision>` plus a receipt in
`$HOME/atlas-install`. A cold build downloads the CUDA base images and
compiles FlashInfer's sparse-MLA prefill, FlashKDA, CUTLASS-based kernels and
the engine; allow well over half an hour (the engine's kernels alone are
~18 minutes). With a warm BuildKit cache a rebuild took about 5 minutes on our
Sparks. Set `ATLAS_BUILD_JOBS` (default 4) to trade speed for memory.

## 4. Convert the checkpoint overlay (once per node, about a minute, needs the GPU)

Atlas serves the NVIDIA checkpoint through an overlay that quantizes the MTP
matrices it needs; everything else is linked back to the original files.

```sh
IMAGE=lloom/atlas-sparkglm:$(git rev-parse HEAD)
SRC="$MODEL_ROOT/nvidia--GLM-5.3-Flash-NVFP4"
OVERLAY="$MODEL_ROOT/atlas-overlay"          # must not exist yet
docker run --rm --gpus=all --user "$(id -u):$(id -g)" \
  -v "$SRC:$SRC:ro" -v "$MODEL_ROOT:$MODEL_ROOT" \
  --entrypoint python3 "$IMAGE" /opt/atlas/converter/convert.py \
  --source "$SRC" --output "$OVERLAY" --library /opt/atlas/converter/libatlas_mtp_quantize.so
docker run --rm --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$MODEL_ROOT:$MODEL_ROOT" --entrypoint python3 "$IMAGE" \
  /opt/atlas/converter/convert.py --source "$SRC" --output "$OVERLAY" --verify-overlay
```

## 5. Start the pair (worker first)

On the node that will be rank **1** (worker), then on rank **0** (leader).
`--leader-address` is rank 0's address on the direct fabric;
`--fabric-interface` is the fabric interface on the node running the command.

```sh
bash research/atlas/install/start-node.sh --rank 1 --leader-address 192.0.2.1 \
  --fabric-interface enp1s0f0np0 --model-root "$MODEL_ROOT" \
  --overlay "$MODEL_ROOT/atlas-overlay" --image "$IMAGE"
# then, on rank 0:
bash research/atlas/install/start-node.sh --rank 0 --leader-address 192.0.2.1 \
  --fabric-interface enp1s0f0np0 --model-root "$MODEL_ROOT" \
  --overlay "$MODEL_ROOT/atlas-overlay" --image "$IMAGE"
until curl -sf http://127.0.0.1:8893/health; do sleep 10; done   # on rank 0
```

Loading takes about two and a half minutes. The very first start and the first
requests of each shape also fill the CUDA kernel cache (`~/.cache/atlas-cuda`),
which later starts reuse; run one warmup pass before measuring. The OpenAI
compatible API listens on rank 0's loopback, `127.0.0.1:8893`, model
`glm-5.3-flash-atlas`; put your own authenticated proxy in front of it before
exposing it. Stop with `docker rm -f atlas-sparkglm-rank0` (and `-rank1`).

Smoke test:

```sh
curl -s http://127.0.0.1:8893/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash-atlas", "max_tokens": 64, "temperature": 0,
  "messages": [{"role": "user", "content": "What is 17*23? Answer with the number."}]}'
```

The engine supports streaming, tool calls (`poolside_v1` parser), JSON/structured
output, reasoning controls (`chat_template_kwargs.enable_thinking`,
`reasoning_effort`), images and short videos.

## 6. Reproduce our numbers

All drivers are in [`reproduce/`](reproduce/) and the repository's
[`benchmarks/`](../../../benchmarks/). Run them on rank 0 against the running
engine, one workload at a time, with nothing else using the pair.

```sh
cd research/atlas/install/reproduce
# Matrix: three passes (compare the sum of the five walls to 152.9 s).
for p in 1 2 3; do MATRIX_SALT=mb-vx2-7c31 python3 run_matrix.py out-matrix-$p matrix; done
# Staggered C4 field guide: warmup + three runs, fresh cache salt.
MATRIX_SALT=fg-$(date +%s) python3 run_matrix.py out-field field
# Hard quality probe.
python3 quality_probe.py mine 4
# mmastrac's prefill/decode benchmark and 1/2/4 streams.
python3 mmastrac_bench.py && python3 mmastrac_streams.py
```

`run_matrix.py` prints one JSON line per case; `summary.wall_s` is the number
we sum. The 8-stream and RigMark results use the 8 × 128K profile: restart both
ranks with `--profile research/atlas/install/profile-8x128k.json`, then run
[RigMark](https://github.com/othexmr/rigmark) at commit `40fabcaf` with the
cell set in [`reproduce/RIGMARK.md`](reproduce/RIGMARK.md).

Comparison video (our field guide against any other recipe's captures):

```sh
python3 build_comparison.py cmp.json \
  "Atlas|this recipe|out-field/field-guide-r1.json,out-field/field-guide-r2.json,out-field/field-guide-r3.json" \
  "Other recipe|its settings|a1.json,a2.json,a3.json"
python3 render_comparison.py cmp.json atlas-c4.mp4 "Atlas vs other · staggered C4"
```

The vLLM and Mia field-guide captures we compared against are in the result
bundle's `raw/` directory.

## Alternative: LLooM

[LLooM](https://github.com/Enntity/lloom) (branch `atlas/installable-20260929`)
automates steps 2–5 with the recipe `linux-nvidia-dgx-spark-2x-glm53-atlas`:

```sh
lloom setup --recipe linux-nvidia-dgx-spark-2x-glm53-atlas --additive --apply --yes
lloom runtime-start glm53-flash-atlas-cluster
```

It pins this same SparkGLM revision, builds the same image on each node,
converts and verifies the overlay, and starts worker then leader behind its
authenticated gateway.

## Troubleshooting

- **NCCL init fails with `modify_qp -> RTR failed`:** the RoCE link or its IPv4
  address is missing on one node. The engine picks each rail's RoCE v2 IPv4 GID
  automatically; to force one, add `"ATLAS_RDMA_GID": "<index>"` to the
  profile's `environment` (the launcher ignores ambient `ATLAS_*` variables).
- **The host becomes unresponsive under load:** GB10 memory is unified; keep
  builds, other models and large processes off the pair while serving. The
  profile leaves ~8–9 GB free at full KV.
- **`thinking` behaviour:** requests without thinking still get a short
  low-effort reasoning block (returned in `reasoning_content`); the reasoning
  budget is enforced, so structured answers are not trapped in reasoning.

## Licensing

The engine and everything under `research/atlas/` are AGPL-3.0-only.
FlashKDA (MIT), FlashInfer (Apache-2.0), CUTLASS (BSD-3) and NVIDIA's sparse-MLA
source keep their licenses; notices ship inside the image under
`/opt/atlas/notices/`. See [`docs/LICENSING.md`](../../../docs/LICENSING.md).
