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
usage: start-node.sh --rank 0-3 --leader-address IP --model-root DIR --image TAG
                     [--fabric-interface IFACE] [--fabric-hca DEVICE[,DEVICE...]]
                     [--world-size 2|4 [--left-hca DEVICE[,...]] [--right-hca DEVICE[,...]]]
                     [--mode strict|spark] [--profile 4x512k|8x128k|FILE] [--gpu-memory-utilization 0.80-0.95 (0.60- on four Sparks)]
                     [--prefix-cache-dir DIR [--prefix-cache-gb 16-100]]
                     [--display-carveout] [--kv-shard] [--no-fabric-siblings] [--checkpoint NAME]
                     [--cuda-cache DIR] [--name NAME]

  --leader-address   rank 0's IPv4 address on the direct Spark-to-Spark fabric
  --model-root       directory holding the checkpoint and
                     incoai--GLM-5.3-Flash-DFlash2
  --checkpoint       the checkpoint's directory under --model-root (default:
                     the pinned nvidia--GLM-5.3-Flash-FP4-Local, served as
                     published); nvidia--GLM-5.3-Flash-NVFP4 is served through
                     the atlas-overlay convert.sh writes beside it
  --image            the image tag, install/build.sh --tag
  --fabric-interface the fabric's network interface on THIS node (default: the
                     interface of the first --fabric-hca; otherwise required)
  --no-fabric-siblings  compatibility option; explicit HCA lists are always
                     passed without adding another device
  --world-size       Sparks serving together (default 2). With 4, wired as a
                     daisy chain rank 0-1-2-3, start ranks 3, 2, 1, then 0;
                     --rank goes up to 3, --fabric-interface is this node's
                     interface for NCCL's startup traffic (every Spark must
                     reach the others' addresses on it; --leader-address is
                     rank 0's address there) and replaces --fabric-hca
  --left-hca         with --world-size 4: the RDMA devices cabled to rank - 1
  --right-hca        with --world-size 4: the RDMA devices cabled to rank + 1
  --mode             strict (default) or spark; TP2 only, same on both ranks
  --strict-mtp       experimental native MTP in strict mode; C1, context 8192
  --profile          a profile shipped in the image (default 4x512k) or a JSON
                     file of your own; use the same one on both ranks
  --gpu-memory-utilization  share of unified memory for the engine (default:
                     the profile's: 0.88, or 0.72 on four Sparks); raise it on dedicated hosts for a
                     larger KV pool and prefix cache
  --prefix-cache-dir keep evicted prefix-cache entries in this directory on
                     the node's own disk (ext4/xfs) instead of dropping them
                     (default: off)
  --prefix-cache-gb  their disk budget in GiB (default 48); the same on both ranks
  --display-carveout lend the GB10's 2 GiB display carveout to the KV cache
                     (2 GiB more KV cache); use it on both ranks (default: off)
  --kv-shard         split the KV cache between the two Sparks, about 1.8x the
                     pool for the same memory; use it on both ranks (default: off)
  --cuda-cache       persistent CUDA JIT cache (default: ~/.cache/atlas-cuda)
EOF
  exit 2
}
rank="" leader="" iface="" model_root="" image="" hca="" world=2 left_hca="" right_hca=""
cache="$HOME/.cache/atlas-cuda" name="" profile=4x512k util="" pc_dir="" pc_gb="" carveout="" siblings=0 kvshard=0
mode=""
mode_given=0
strict_mtp=0
checkpoint_name=nvidia--GLM-5.3-Flash-FP4-Local
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank) rank=$2; shift 2 ;;
    --leader-address) leader=$2; shift 2 ;;
    --fabric-interface) iface=$2; shift 2 ;;
    --fabric-hca) hca=$2; shift 2 ;;
    --world-size) world=$2; shift 2 ;;
    --left-hca) left_hca=$2; shift 2 ;;
    --right-hca) right_hca=$2; shift 2 ;;
    --model-root) model_root=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --cuda-cache) cache=$2; shift 2 ;;
    --checkpoint) checkpoint_name=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    --mode) mode=$2; mode_given=1; shift 2 ;;
    --strict-mtp) strict_mtp=1; shift ;;
    --gpu-memory-utilization) util=$2; shift 2 ;;
    --prefix-cache-dir) pc_dir=$2; shift 2 ;;
    --prefix-cache-gb) pc_gb=$2; shift 2 ;;
    --display-carveout) carveout=1; shift ;;
    --kv-shard) kvshard=1; shift ;;
    --no-fabric-siblings) siblings=0; shift ;;
    *) usage ;;
  esac
done
[[ $world == 2 || $world == 4 ]] && [[ $rank =~ ^[0-9]$ ]] && (( rank < world )) &&
  [[ -n $leader && -n $model_root && -n $image ]] || usage
if [[ $world == 2 ]]; then
  if [[ $mode_given == 0 ]]; then mode=strict; fi
  [[ $mode == strict || $mode == spark ]] || { echo "--mode must be strict or spark" >&2; exit 2; }
else
  [[ -z $mode ]] || { echo "strict/spark modes require TP2" >&2; exit 2; }
fi
sysfs=${FABRIC_SYSFS:-/sys/class/infiniband}
# check_rails LIST: every device named once, up and well formed.
check_rails() {
  [[ $1 =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*(,[a-zA-Z0-9_][a-zA-Z0-9_.-]*)*$ ]] ||
    { echo "invalid RDMA device list $1" >&2; exit 2; }
  local rail seen=,
  IFS=',' read -r -a rails <<< "$1"
  for rail in "${rails[@]}"; do
    [[ $seen != *,"$rail",* ]] || { echo "duplicate RDMA device $rail" >&2; exit 2; }
    seen+="$rail,"
    grep -qs ACTIVE "$sysfs/$rail/ports/1/state" ||
      { echo "RDMA device $rail is missing or down" >&2; exit 2; }
  done
}
if (( world > 2 )); then
  [[ -z $hca ]] || { echo "a chain takes --left-hca and --right-hca, not --fabric-hca" >&2; exit 2; }
  [[ $iface =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || { echo "a chain needs --fabric-interface" >&2; exit 2; }
  if (( rank > 0 )); then [[ -n $left_hca ]] || { echo "rank $rank needs --left-hca" >&2; exit 2; }
  else [[ -z $left_hca ]] || { echo "rank 0 has no left neighbour" >&2; exit 2; }; fi
  if (( rank < world - 1 )); then [[ -n $right_hca ]] || { echo "rank $rank needs --right-hca" >&2; exit 2; }
  else [[ -z $right_hca ]] || { echo "rank $rank has no right neighbour" >&2; exit 2; }; fi
  for list in "$left_hca" "$right_hca"; do [[ -z $list ]] || check_rails "$list"; done
else
  [[ -z $left_hca && -z $right_hca ]] || { echo "--left-hca and --right-hca need --world-size 4" >&2; exit 2; }
  if [[ -z $hca ]]; then
    [[ -n $iface ]] || { echo "pass --fabric-hca or --fabric-interface; no port is assumed" >&2; exit 2; }
    matches=()
    for dev in "$sysfs"/*; do
      [[ -e $dev/device/net/$iface ]] && matches+=("${dev##*/}")
    done
    [[ ${#matches[@]} == 1 ]] || { echo "ambiguous or missing RDMA device for $iface; pass --fabric-hca" >&2; exit 2; }
    hca=${matches[0]}
  fi
  check_rails "$hca"
  if [[ -z $iface ]]; then
    primary=${hca%%,*}
    matches=("$sysfs/$primary"/device/net/*)
    [[ ${#matches[@]} == 1 && -e ${matches[0]} ]] ||
      { echo "ambiguous or missing network interface for $primary; pass --fabric-interface" >&2; exit 2; }
    iface=${matches[0]##*/}
  fi
  [[ $iface =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ && -e $sysfs/${hca%%,*}/device/net/$iface ]] ||
    { echo "--fabric-interface must belong to the first --fabric-hca device" >&2; exit 2; }
fi
[[ $checkpoint_name =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] ||
  { echo "--checkpoint must be one directory name under --model-root" >&2; exit 2; }
model_root=$(realpath "$model_root")
checkpoint="$model_root/$checkpoint_name"
drafter="$model_root/incoai--GLM-5.3-Flash-DFlash2"
model=$checkpoint
model_mounts=(--mount "type=bind,src=$checkpoint,dst=$checkpoint,readonly")
required=("$checkpoint/config.json" "$drafter/config.json")
# NVIDIA's NVFP4 release ships its MTP layer in BF16; Atlas serves it through
# the overlay convert.sh writes. The overlay links into the original
# checkpoint, so both are mounted at the same absolute paths.
if [[ $checkpoint_name == nvidia--GLM-5.3-Flash-NVFP4 ]]; then
  model="$model_root/atlas-overlay"
  model_mounts+=(--mount "type=bind,src=$model,dst=$model,readonly")
  required+=("$model/sparkglm-verified.json")
fi
for path in "${required[@]}"; do
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
if [[ -n $pc_dir || -n $pc_gb ]]; then
  pc_gb=${pc_gb:-48}
  [[ $pc_dir == /* && $pc_dir != *,* ]] || { echo "--prefix-cache-dir must be set to an absolute path without commas" >&2; exit 2; }
  [[ $pc_gb =~ ^[1-9][0-9]{1,2}$ ]] && (( pc_gb >= 16 && pc_gb <= 100 )) ||
    { echo "--prefix-cache-gb must be a whole number from 16 to 100" >&2; exit 2; }
fi
name=${name:-atlas-sparkglm-rank$rank}
# The engine unlinks its cache files while they are open and sweeps any a crash
# left behind, so nothing carries over a restart.
pc_args=()
if [[ -n $pc_dir ]]; then
  mkdir -p "$pc_dir/kv" "$pc_dir/ssm"
  pc_dir=$(realpath "$pc_dir")
  case $(stat -f -c %T "$pc_dir" 2>/dev/null) in
    tmpfs | ramfs | overlayfs) echo "--prefix-cache-dir $pc_dir is not on a disk" >&2; exit 2 ;;
  esac
  free_kb=$(df -Pk "$pc_dir" | awk 'NR == 2 {print $4}')
  (( free_kb >= pc_gb * 1048576 )) ||
    { echo "--prefix-cache-dir $pc_dir has $((free_kb / 1048576)) GiB free; --prefix-cache-gb needs $pc_gb" >&2; exit 2; }
  pc_args=(--mount "type=bind,src=$pc_dir,dst=/prefix-cache" -e SPARKGLM_PREFIX_CACHE_GB="$pc_gb")
fi
# One process per host may hold the carveout; every container shares this
# host directory's lock file, which the server holds while it runs.
carveout_args=()
if [[ -n $carveout ]]; then
  lock_dir=/run/lock/sparkglm
  mkdir -p "$lock_dir" && chmod 1777 "$lock_dir" 2>/dev/null || true
  carveout_args=(--mount "type=bind,src=$lock_dir,dst=$lock_dir"
    -e SPARKGLM_DISPLAY_CARVEOUT=1)
fi
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --restart no --network host --ipc host \
  --shm-size 32g --memory 114g --gpus all --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
  --security-opt no-new-privileges=true --stop-timeout 60 \
  "${model_mounts[@]}" \
  --mount "type=bind,src=$drafter,dst=$drafter,readonly" \
  --mount "type=bind,src=$cache,dst=/atlas-cuda-cache" ${profile_mount[@]+"${profile_mount[@]}"} ${pc_args[@]+"${pc_args[@]}"} \
  ${carveout_args[@]+"${carveout_args[@]}"} \
  -e NODE_RANK="$rank" -e MASTER_ADDR="$leader" -e MASTER_PORT=29510 \
  -e FABRIC_INTERFACE="$iface" -e FABRIC_HCA="$hca" -e FABRIC_SIBLINGS="$siblings" -e SPARKGLM_KV_SHARD="$kvshard" \
  -e FABRIC_LEFT_HCA="$left_hca" -e FABRIC_RIGHT_HCA="$right_hca" \
  -e MODEL_PATH="$model" -e DRAFTER_PATH="$drafter" \
  -e SPARKGLM_STRICT_MTP="$strict_mtp" \
  -e SERVED_MODEL_NAME=glm-5.3-flash-atlas -e SPARKGLM_PROFILE="$profile" ${mode:+-e SPARKGLM_MODE="$mode"} \
  ${util:+-e SPARKGLM_GPU_MEMORY_UTILIZATION="$util"} \
  -e ATLAS_WORLD_SIZE="$world" -e ATLAS_TP_SIZE="$world" -e ATLAS_EP_SIZE="$world" -e ATLAS_CONTEXT_WINDOW=524288 \
  -e NCCL_SOCKET_IFNAME="$iface" -e GLOO_SOCKET_IFNAME="$iface" -e NCCL_IB_HCA="$hca" \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_CROSS_NIC=0 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_RETRY_CNT=7 -e NCCL_IB_TIMEOUT=22 \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_BUFFSIZE=33554432 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_MAX_NCHANNELS=2 -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_DMABUF_ENABLE=0 -e NCCL_DEBUG=WARN -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_CACHE_PATH=/atlas-cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
  "$image"
echo "started $name (rank $rank); follow with: docker logs -f $name"
