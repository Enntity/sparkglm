#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SparkGLM: GLM-5.3-Flash on two DGX Sparks, served by Atlas. Run on the Spark
# that will serve the API (rank 0); the other one (rank 1) is driven over ssh.
#
#   ./start.sh              prepare whatever is missing, then serve
#   ./start.sh stop         stop both ranks
#   ./start.sh status       container state and health
#   ./start.sh logs [worker]
#   ./start.sh build | download | convert    run one preparation step
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${WORKER:?set WORKER in .env to the ssh destination of the other Spark (see .env.example)}"
MODEL_ROOT=${MODEL_ROOT:-$HOME/models/sparkglm}
PROFILE=${PROFILE:-4x512k}
FABRIC_HCA=${FABRIC_HCA:-rocep1s0f0}
IMAGE=${IMAGE:-$(install/build.sh --tag)}
CHECKPOINT=nvidia/GLM-5.3-Flash-NVFP4@423acf37583782c51c142d145aef733d72943d93
DRAFTER=incoai/GLM-5.3-Flash-DFlash2@7d74cdd881ed7e32c31175984a67823127b66cfe
API=http://127.0.0.1:8893
NAME=atlas-sparkglm-rank

say() { printf '\033[1m== %s\033[0m\n' "$*"; }
on_worker() { ssh -o BatchMode=yes "$WORKER" "$@"; }
# Run a local script on the worker with safely quoted arguments.
script_on_worker() { local script=$1; shift; on_worker "bash -s -- $(printf '%q ' "$@")" < "$script"; }
has_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

build() {
  say "image $IMAGE"
  if ! has_image; then
    if [[ ${BUILD:-0} == 1 ]] || ! docker pull "$IMAGE"; then
      say "building $IMAGE from source (a cold build takes 30-60 minutes)"
      IMAGE=$(install/build.sh)
    fi
  fi
  if ! on_worker docker image inspect "$IMAGE" >/dev/null 2>&1; then
    on_worker docker pull "$IMAGE" || { say "copying the image to $WORKER"; docker save "$IMAGE" | on_worker docker load; }
  fi
}

hf_download() {  # repo revision dir; uses a throwaway container when hf is not installed
  if command -v hf >/dev/null; then
    hf download "$1" --revision "$2" --local-dir "$3"
  else
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN -v "$MODEL_ROOT:$MODEL_ROOT" \
      python:3.12-slim sh -c 'pip install -q --disable-pip-version-check --target /tmp/hf huggingface_hub &&
        PYTHONPATH=/tmp/hf /tmp/hf/bin/hf download "$0" --revision "$1" --local-dir "$2"' "$1" "$2" "$3"
  fi
}

# One ssh stream moves ~0.5 GB/s; eight parallel streams fill ~3.5 GB/s of the
# cable. The final rsync picks up the small files and anything left over.
copy_to_worker() {  # directory name under MODEL_ROOT
  on_worker mkdir -p "$MODEL_ROOT"
  (cd "$MODEL_ROOT" && find "$1" -type f -size +64M ! -name '*.incomplete' -print0) |
    xargs -0 -P8 -I{} rsync -aR -e 'ssh -c aes128-gcm@openssh.com' "$MODEL_ROOT/./{}" "$WORKER:$MODEL_ROOT/"
  rsync -a --exclude "*.incomplete" "$MODEL_ROOT/$1" "$WORKER:$MODEL_ROOT/"
}

# Download each pinned checkpoint once, then mirror it to the worker. The
# marker is written only after both copies are complete.
download() {
  local spec repo revision dir
  for spec in "$CHECKPOINT" "$DRAFTER"; do
    repo=${spec%@*} revision=${spec#*@}
    dir="$MODEL_ROOT/${repo/\//--}"
    [[ $(cat "$dir/.sparkglm-revision" 2>/dev/null) == "$revision" ]] && continue
    say "download $repo @ $revision"
    hf_download "$repo" "$revision" "$dir"
    say "copy $repo to $WORKER:$MODEL_ROOT"
    copy_to_worker "${dir##*/}"
    echo "$revision" > "$dir/.sparkglm-revision"
  done
}

convert() {
  say "overlay on this node"; install/convert.sh "$MODEL_ROOT" "$IMAGE"
  say "overlay on $WORKER"; script_on_worker install/convert.sh "$MODEL_ROOT" "$IMAGE"
}

stop() {
  docker rm -f "${NAME}0" >/dev/null 2>&1 || true
  on_worker docker rm -f "${NAME}1" >/dev/null 2>&1 || true
}

state() { docker inspect -f '{{.State.Status}}' "${NAME}0" 2>/dev/null || echo absent; }
worker_state() { on_worker docker inspect -f '{{.State.Status}}' "${NAME}1" 2>/dev/null || echo absent; }

serve() {
  local iface address common
  iface=${FABRIC_INTERFACE:-$(ls "/sys/class/infiniband/$FABRIC_HCA/device/net" 2>/dev/null | head -1)}
  address=${LEADER_ADDRESS:-}
  [[ -n $address ]] || address=$(ip -4 -o addr show dev "$iface" 2>/dev/null | awk '{sub("/.*", "", $4); print $4; exit}')
  [[ -n $address ]] || { echo "no IPv4 address on the fabric interface '$iface'; set LEADER_ADDRESS" >&2; exit 2; }
  common=(--leader-address "$address" --model-root "$MODEL_ROOT" --image "$IMAGE"
          --profile "$PROFILE" --fabric-hca "$FABRIC_HCA" ${FABRIC_INTERFACE:+--fabric-interface "$FABRIC_INTERFACE"}
          ${GPU_MEMORY_UTILIZATION:+--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"}
          ${PREFIX_CACHE_DIR:+--prefix-cache-dir "$PREFIX_CACHE_DIR"} ${PREFIX_CACHE_GB:+--prefix-cache-gb "$PREFIX_CACHE_GB"})
  case ${DISPLAY_CARVEOUT:-0} in
    0) ;;
    1) common+=(--display-carveout) ;;
    *) echo "DISPLAY_CARVEOUT must be 0 or 1" >&2; exit 2 ;;
  esac
  case ${KV_SHARD:-0} in
    0) ;;
    1) [[ -z ${PREFIX_CACHE_DIR:-} ]] || { echo "KV_SHARD=1 and PREFIX_CACHE_DIR cannot be combined; unset one" >&2; exit 2; }
       common+=(--kv-shard) ;;
    *) echo "KV_SHARD must be 0 or 1" >&2; exit 2 ;;
  esac
  case ${FABRIC_SIBLINGS:-1} in
    1) ;;
    0) common+=(--no-fabric-siblings) ;;
    *) echo "FABRIC_SIBLINGS must be 0 or 1" >&2; exit 2 ;;
  esac
  say "start rank 1 on $WORKER, then rank 0 here (leader $address, profile $PROFILE)"
  stop
  script_on_worker install/start-node.sh --rank 1 "${common[@]}"
  install/start-node.sh --rank 0 "${common[@]}"
  say "loading (about 2.5 minutes)"
  for ((i = 0; i < 90; i++)); do
    if curl -sf "$API/health" >/dev/null; then
      say "ready: $API/v1 model glm-5.3-flash-atlas"
      curl -s "$API/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model": "glm-5.3-flash-atlas",
        "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": "What is 17*23? Answer with the number."}]}' |
        python3 -c 'import json,sys; print("smoke: 17*23 =", json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'
      return
    fi
    if [[ $(state) != running ]]; then docker logs --tail 40 "${NAME}0"; echo 'rank 0 stopped' >&2; exit 1; fi
    if [[ $(worker_state) != running ]]; then on_worker docker logs --tail 40 "${NAME}1"; echo 'rank 1 stopped' >&2; exit 1; fi
    sleep 10
  done
  echo 'not ready after 15 minutes; see ./start.sh logs' >&2; exit 1
}

mkdir -p "$MODEL_ROOT"; MODEL_ROOT=$(realpath "$MODEL_ROOT")
case ${1:-up} in
  up) build; download; convert; serve ;;
  build | download | convert | stop) "$1" ;;
  status)
    echo "image  $IMAGE"
    echo "rank 0 $(state)"; echo "rank 1 $(worker_state) ($WORKER)"
    curl -sf "$API/health" >/dev/null && echo "health ok ($API)" || echo 'health: not serving' ;;
  logs) if [[ ${2:-} == worker ]]; then on_worker docker logs -f --tail 100 "${NAME}1"; else docker logs -f --tail 100 "${NAME}0"; fi ;;
  *) sed -n '5,11p' "$0" >&2; exit 2 ;;
esac
