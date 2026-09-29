#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Start one rank of the Atlas SparkGLM image without LLooM. Start rank 1 (the
# worker) first, then rank 0 (the leader, which serves the OpenAI API on
# 127.0.0.1:8893). Container flags, mounts and environment match the LLooM
# recipe linux-nvidia-dgx-spark-2x-glm53-atlas; this script only starts the
# prepared image and never builds, downloads or converts anything.
set -euo pipefail
usage() {
  cat >&2 <<'EOF'
usage: start-node.sh --rank 0|1 --leader-address IP --model-root DIR --image TAG
                     [--fabric-interface IFACE] [--fabric-hca rocep1s0f0]
                     [--profile 4x512k|8x128k|FILE] [--gpu-memory-utilization 0.80-0.95]
                     [--cuda-cache DIR] [--name NAME]

  --leader-address   rank 0's IPv4 address on the direct Spark-to-Spark fabric
  --model-root       directory holding nvidia--GLM-5.3-Flash-NVFP4,
                     incoai--GLM-5.3-Flash-DFlash2 and atlas-overlay (convert.sh)
  --image            the image tag, install/build.sh --tag
  --fabric-interface the fabric's network interface on THIS node (default: the
                     interface of --fabric-hca, e.g. enp1s0f0np0)
  --profile          a profile shipped in the image (default 4x512k) or a JSON
                     file of your own; use the same one on both ranks
  --gpu-memory-utilization  share of unified memory for the engine (default:
                     the profile's 0.88); raise it on dedicated hosts for a
                     larger KV pool and prefix cache
  --cuda-cache       persistent CUDA JIT cache (default: ~/.cache/atlas-cuda)
EOF
  exit 2
}
rank="" leader="" iface="" model_root="" image="" hca=rocep1s0f0
cache="$HOME/.cache/atlas-cuda" name="" profile=4x512k util=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank) rank=$2; shift 2 ;;
    --leader-address) leader=$2; shift 2 ;;
    --fabric-interface) iface=$2; shift 2 ;;
    --fabric-hca) hca=$2; shift 2 ;;
    --model-root) model_root=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --cuda-cache) cache=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    --gpu-memory-utilization) util=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ $rank == 0 || $rank == 1 ]] && [[ -n $leader && -n $model_root && -n $image ]] || usage
if [[ -z $iface ]]; then
  iface=$(ls "/sys/class/infiniband/$hca/device/net" 2>/dev/null | head -1)
  [[ -n $iface ]] || { echo "no network interface for $hca; pass --fabric-interface" >&2; exit 2; }
fi
model_root=$(realpath "$model_root")
overlay="$model_root/atlas-overlay"
checkpoint="$model_root/nvidia--GLM-5.3-Flash-NVFP4"
drafter="$model_root/incoai--GLM-5.3-Flash-DFlash2"
for path in "$checkpoint/config.json" "$drafter/config.json" "$overlay/sparkglm-verified.json"; do
  [[ -f $path ]] || { echo "missing $path" >&2; exit 2; }
done
mkdir -p "$cache"
profile_mount=()
if [[ $profile == */* || $profile == *.json ]] && [[ ! -f $profile ]]; then
  echo "missing profile file $profile" >&2; exit 2
fi
if [[ -f $profile ]]; then
  profile_mount=(--mount "type=bind,src=$(realpath "$profile"),dst=/opt/atlas/profiles/custom.json,readonly")
  profile=custom
fi
name=${name:-atlas-sparkglm-rank$rank}
docker rm -f "$name" >/dev/null 2>&1 || true
# The overlay links into the original checkpoint, so it is mounted at the same
# absolute path inside the container.
docker run -d --name "$name" --restart no --network host --ipc host \
  --shm-size 32g --memory 114g --gpus all --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
  --security-opt no-new-privileges=true --stop-timeout 60 \
  --mount "type=bind,src=$checkpoint,dst=$checkpoint,readonly" \
  --mount "type=bind,src=$overlay,dst=$overlay,readonly" \
  --mount "type=bind,src=$drafter,dst=$drafter,readonly" \
  --mount "type=bind,src=$cache,dst=/atlas-cuda-cache" "${profile_mount[@]}" \
  -e NODE_RANK="$rank" -e MASTER_ADDR="$leader" -e MASTER_PORT=29510 \
  -e FABRIC_INTERFACE="$iface" -e FABRIC_HCA="$hca" \
  -e MODEL_PATH="$overlay" -e DRAFTER_PATH="$drafter" \
  -e SERVED_MODEL_NAME=glm-5.3-flash-atlas -e SPARKGLM_PROFILE="$profile" \
  ${util:+-e SPARKGLM_GPU_MEMORY_UTILIZATION="$util"} \
  -e ATLAS_WORLD_SIZE=2 -e ATLAS_TP_SIZE=2 -e ATLAS_EP_SIZE=2 -e ATLAS_CONTEXT_WINDOW=524288 \
  -e NCCL_SOCKET_IFNAME="$iface" -e GLOO_SOCKET_IFNAME="$iface" -e NCCL_IB_HCA="$hca" \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_CROSS_NIC=0 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_RETRY_CNT=7 -e NCCL_IB_TIMEOUT=22 \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_BUFFSIZE=33554432 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_MAX_NCHANNELS=2 -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_DMABUF_ENABLE=0 -e NCCL_DEBUG=WARN -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_CACHE_PATH=/atlas-cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
  "$image"
echo "started $name (rank $rank); follow with: docker logs -f $name"
