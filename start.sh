#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SparkGLM: GLM-5.3-Flash on two or four DGX Sparks, served by Atlas. Run on the
# Spark that will serve the API (rank 0); WORKER names the others, driven over ssh.
#
#   ./start.sh              prepare whatever is missing, then serve
#   ./start.sh fabric       inspect the selected ports without changing anything
#   ./start.sh clock-cap    cap the GPU clock on every Spark (sudo; once per Spark)
#   ./start.sh --version    checkout product version and pinned engine
#   ./start.sh stop         stop every rank
#   ./start.sh status       container state and health
#   ./start.sh logs [worker [HOST]]
#   ./start.sh build | download | convert    run one preparation step
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ${1:-} == --version ]]; then exec python3 install/serve.py --version; fi
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${WORKER:?set WORKER in .env to the ssh destination of the other Spark, or of the other three (see .env.example)}"
read -r -a workers <<< "$WORKER"
case ${#workers[@]} in
  1) world=2 tp_profile=4x1m ;;
  3) world=4 tp_profile=4x1m-tp4 ;;
  *) echo "WORKER names the other Spark, or the other three of four" >&2; exit 2 ;;
esac
MODEL_ROOT=${MODEL_ROOT:-$HOME/models/sparkglm}
PROFILE=${PROFILE:-$tp_profile}
MODE=${MODE:-strict}
if [[ ${1:-} == --mode ]]; then MODE=${2:-}; shift 2; fi
[[ $MODE == strict || $MODE == spark ]] || { echo "--mode must be strict or spark" >&2; exit 2; }
IMAGE=${IMAGE:-}
CHECKPOINT=nvidia/GLM-5.3-Flash-FP4-Local@76348ddb2be2e35e39ecfc00e75dc07703f4f3d5
# .env may name another checkpoint (both values or neither); the image stays the same.
if [[ -n ${CHECKPOINT_REPO:-} || -n ${CHECKPOINT_REVISION:-} ]]; then
  [[ ${CHECKPOINT_REPO:-} =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$ && ${CHECKPOINT_REVISION:-} =~ ^[0-9a-f]{40}$ ]] ||
    { echo 'set both CHECKPOINT_REPO (owner/name) and CHECKPOINT_REVISION (a 40-hex commit) in .env, or neither' >&2; exit 2; }
  CHECKPOINT=$CHECKPOINT_REPO@$CHECKPOINT_REVISION
fi
DRAFTER=incoai/GLM-5.3-Flash-DFlash2@6cc6c481eee930a5139619ed919f78726371adbc
API=http://127.0.0.1:8893
NAME=atlas-sparkglm-rank

say() { printf '\033[1m== %s\033[0m\n' "$*"; }
on() { ssh -o BatchMode=yes "$1" "${@:2}"; }
# Run a local script on a worker with safely quoted arguments.
script_on() { local host=$1 script=$2; shift 2; on "$host" "bash -s -- $(printf '%q ' "$@")" < "$script"; }
has_image() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

fabric_ports() { bash install/fabric-ports.sh; }

pick_fabric() {
  local here there choice
  ((world == 4)) && { pick_ring; return; }
  here=$(fabric_ports)
  there=$(on "${workers[0]}" "bash -s" < install/fabric-ports.sh)
  choice=$(python3 fabric.py "$here" "$there" \
    "${LEADER_FABRIC_HCA:-${FABRIC_HCA:-}}" "${WORKER_FABRIC_HCA:-${FABRIC_HCA:-}}" \
    "${LEADER_FABRIC_INTERFACE:-${FABRIC_INTERFACE:-}}" "${WORKER_FABRIC_INTERFACE:-${FABRIC_INTERFACE:-}}" \
    "${LEADER_ADDRESS:-}") || return 2
  read -r hca0 iface0 ip0 hca1 iface1 ip1 <<< "$choice"
  say "fabric: $hca0 ($iface0, $ip0) here; $hca1 ($iface1, $ip1) on ${workers[0]}"
}

# Four Sparks: one line per rank (rank host kind interface leader left right),
# host 0 being this one, in the order the cables put them.
pick_ring() {
  local here inventories=() host rank kind iface left right
  here=$(fabric_ports)
  for host in "${workers[@]}"; do inventories+=("$(on "$host" "bash -s" < install/fabric-ports.sh)"); done
  plan=$(python3 fabric.py topology "$here" "${inventories[@]}") || return 2
  while read -r rank host kind iface _ left right; do
    say "rank $rank on $( ((host)) && echo "${workers[host - 1]}" || echo here): $kind, $iface, left $left, right $right"
  done <<< "$plan"
}

# Sustained full GPU load can power an uncapped Spark off, with no log.
clock_cap() {
  local max=${GPU_CLOCK_MAX_MHZ:-2200}
  # It reaches a root shell on the worker unquoted, so it must be a plain number.
  [[ $max =~ ^[0-9]+$ ]] || { echo "GPU_CLOCK_MAX_MHZ must be a number of MHz" >&2; exit 2; }
  say "GPU clock cap 300,$max MHz here (sudo)"
  sudo bash gpu-clock-cap.sh "$max"
  for host in "${workers[@]}"; do
    say "GPU clock cap 300,$max MHz on $host (sudo)"
    ssh -t "$host" "sudo bash -c $(printf '%q' "$(cat gpu-clock-cap.sh)") gpu-clock-cap $max"
  done
}

check_clock_cap() {
  [[ ${CLOCK_CAP:-1} == 0 ]] && return
  local host
  for host in here "${workers[@]}"; do
    if [[ $host == here ]]; then systemctl is-active --quiet gpu-clock-cap.service && continue
    else on "$host" systemctl is-active --quiet gpu-clock-cap.service && continue; fi
    echo "GPU clocks are not capped on $host: long prefills can power the Spark off." >&2
    echo "Run ./start.sh clock-cap once (sudo on both Sparks), or set CLOCK_CAP=0 to skip this check." >&2
    exit 2
  done
}

build() {
  say "image $IMAGE"
  if ! has_image; then
    if [[ ${BUILD:-0} == 1 ]] || ! docker pull "$IMAGE"; then
      say "building $IMAGE from source (a cold build takes 30-60 minutes)"
      IMAGE=$(install/build.sh)
    fi
  fi
  for host in "${workers[@]}"; do
    on "$host" docker image inspect "$IMAGE" >/dev/null 2>&1 && continue
    on "$host" docker pull "$IMAGE" || { say "copying the image to $host"; docker save "$IMAGE" | on "$host" docker load; }
  done
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
copy_to() {  # worker, directory name under MODEL_ROOT
  on "$1" mkdir -p "$MODEL_ROOT"
  (cd "$MODEL_ROOT" && find "$2" -type f -size +64M ! -name '*.incomplete' -print0) |
    xargs -0 -P8 -I{} rsync -aR -e 'ssh -c aes128-gcm@openssh.com' "$MODEL_ROOT/./{}" "$1:$MODEL_ROOT/"
  rsync -a --exclude "*.incomplete" "$MODEL_ROOT/$2" "$1:$MODEL_ROOT/"
}

# Download each pinned checkpoint once, then mirror it to the workers. The
# marker is written only after every copy is complete.
download() {
  local spec repo revision dir host
  for spec in "$CHECKPOINT" "$DRAFTER"; do
    repo=${spec%@*} revision=${spec#*@}
    dir="$MODEL_ROOT/${repo/\//--}"
    [[ $(cat "$dir/.sparkglm-revision" 2>/dev/null) == "$revision" ]] && continue
    say "download $repo @ $revision"
    hf_download "$repo" "$revision" "$dir"
    for host in "${workers[@]}"; do say "copy $repo to $host:$MODEL_ROOT"; copy_to "$host" "${dir##*/}"; done
    echo "$revision" > "$dir/.sparkglm-revision"
  done
}

# Only NVIDIA's earlier NVFP4 release needs the MTP overlay; FP4-Local ships its
# MTP layer quantized and is served as published.
convert() {
  [[ ${CHECKPOINT%@*} == nvidia/GLM-5.3-Flash-NVFP4 ]] || { say "no overlay: ${CHECKPOINT%@*} is served as published"; return; }
  say "overlay on this node"; install/convert.sh "$MODEL_ROOT" "$IMAGE"
  for host in "${workers[@]}"; do say "overlay on $host"; script_on "$host" install/convert.sh "$MODEL_ROOT" "$IMAGE"; done
}

stop() {
  local host
  docker rm -f "${NAME}0" >/dev/null 2>&1 || true
  for host in "${workers[@]}"; do on "$host" docker rm -f "${NAME}1" "${NAME}2" "${NAME}3" >/dev/null 2>&1 || true; done
}

state() { docker inspect -f '{{.State.Status}}' "${NAME}0" 2>/dev/null || echo absent; }
# A worker holds one rank, whichever the cables gave it.
worker_state() { on "$1" docker ps -a --filter "name=$NAME" --format "'{{.State}}'" 2>/dev/null | grep . || echo absent; }
worker_logs() { on "$1" "docker logs ${*:2} \$(docker ps -aq --filter name=$NAME | head -1)"; }

serve() {
  local common repo=${CHECKPOINT%@*}
  common=(--model-root "$MODEL_ROOT" --checkpoint "${repo/\//--}" --image "$IMAGE"
          --profile "$PROFILE" --mode "$MODE"
          --tool-guard "${TOOL_GUARD:-on}" --tool-guard-max-bytes "${TOOL_GUARD_MAX_BYTES:-1048576}"
          --tool-guard-allow-hosts "${TOOL_GUARD_ALLOW_HOSTS:-}" --tool-guard-deny-hosts "${TOOL_GUARD_DENY_HOSTS:-}"
          ${GPU_MEMORY_UTILIZATION:+--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"}
          ${PREFIX_CACHE_DIR:+--prefix-cache-dir "$PREFIX_CACHE_DIR"} ${PREFIX_CACHE_GB:+--prefix-cache-gb "$PREFIX_CACHE_GB"})
  case ${DISPLAY_CARVEOUT:-0} in
    0) ;;
    1) common+=(--display-carveout) ;;
    *) echo "DISPLAY_CARVEOUT must be 0 or 1" >&2; exit 2 ;;
  esac
  case ${KV_SHARD:-1} in
    0) ;;
    1) common+=(--kv-shard) ;;
    *) echo "KV_SHARD must be 0 or 1" >&2; exit 2 ;;
  esac
  case ${STRICT_MTP:-0} in
    0) ;;
    1) common+=(--strict-mtp) ;;
    *) echo "STRICT_MTP must be 0 or 1" >&2; exit 2 ;;
  esac
  stop
  if ((world == 4)); then
    say "start the ranks on ${workers[*]}, then rank 0 here (profile $PROFILE, mode $MODE)"
    local rank host kind iface leader left right args
    while read -r rank host kind iface leader left right; do
      # TOPOLOGY=chain leaves a ring's closing link unused.
      if [[ ${TOPOLOGY:-$kind} == chain ]]; then
        kind=chain; ((rank == 0)) && left=-; ((rank == 3)) && right=-
      fi
      args=(--rank "$rank" --world-size 4 --topology "$kind" --leader-address "$leader" --fabric-interface "$iface" "${common[@]}")
      [[ $left == - ]] || args+=(--left-hca "$left")
      [[ $right == - ]] || args+=(--right-hca "$right")
      if ((host)); then script_on "${workers[host - 1]}" install/start-node.sh "${args[@]}"
      else install/start-node.sh "${args[@]}"; fi
    done < <(sort -rn <<< "$plan")
  else
    say "start rank 1 on ${workers[0]}, then rank 0 here (leader $ip0, profile $PROFILE, mode $MODE)"
    script_on "${workers[0]}" install/start-node.sh --rank 1 --leader-address "$ip0" --fabric-hca "$hca1" --fabric-interface "$iface1" --no-fabric-siblings "${common[@]}"
    install/start-node.sh --rank 0 --leader-address "$ip0" --fabric-hca "$hca0" --fabric-interface "$iface0" --no-fabric-siblings "${common[@]}"
  fi
  say "loading (about 2.5 minutes)"
  for ((i = 0; i < 120; i++)); do
    if curl -sf "$API/health" >/dev/null; then
      curl -fsS --max-time 300 "$API/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model": "glm-5.3-flash-atlas",
        "max_tokens": 1024, "temperature": 0, "top_p": 1, "reasoning_effort": "low",
        "messages": [{"role": "user", "content": "What is 17*23? Answer with the number."}]}' |
        python3 install/smoke.py
      say "ready: $API/v1 model glm-5.3-flash-atlas"
      return
    fi
    if [[ $(state) != running ]]; then docker logs --tail 40 "${NAME}0"; echo 'rank 0 stopped' >&2; exit 1; fi
    for host in "${workers[@]}"; do
      if [[ $(worker_state "$host") != running ]]; then worker_logs "$host" --tail 40; echo "the rank on $host stopped" >&2; exit 1; fi
    done
    sleep 10
  done
  echo 'not ready after 20 minutes; see ./start.sh logs' >&2; exit 1
}

if [[ ${1:-up} == fabric ]]; then pick_fabric; exit; fi
if [[ ${1:-up} == clock-cap ]]; then clock_cap; exit; fi
if [[ ${1:-up} == up ]]; then
  SPARKGLM_TOOL_GUARD="${TOOL_GUARD:-on}" SPARKGLM_TOOL_GUARD_MAX_BYTES="${TOOL_GUARD_MAX_BYTES:-1048576}" \
  SPARKGLM_TOOL_GUARD_ALLOW_HOSTS="${TOOL_GUARD_ALLOW_HOSTS:-}" SPARKGLM_TOOL_GUARD_DENY_HOSTS="${TOOL_GUARD_DENY_HOSTS:-}" \
    python3 -c 'import os,sys; sys.path.insert(0,"install"); import serve; serve.tool_guard_args(os.environ, [])'
  pick_fabric; check_clock_cap
fi
IMAGE=${IMAGE:-$(install/build.sh --tag)}
mkdir -p "$MODEL_ROOT"; MODEL_ROOT=$(realpath "$MODEL_ROOT")
case ${1:-up} in
  up) build; download; convert; serve ;;
  build | download | convert | stop) "$1" ;;
  status)
    echo "checkout: $(python3 install/serve.py --version | head -1)"
    docker inspect -f 'running rank 0: version={{index .Config.Labels "org.opencontainers.image.version"}} image={{.Image}}' "${NAME}0" 2>/dev/null || true
    for host in "${workers[@]}"; do
      echo "$host: $(on "$host" docker ps -a --filter "name=$NAME" --format "'{{.Names}} {{.State}} image={{.Image}}'" 2>/dev/null)"
    done
    echo "image  $IMAGE"
    echo "rank 0 $(state)"
    curl -sf "$API/health" >/dev/null && echo "health ok ($API)" || echo 'health: not serving' ;;
  logs) if [[ ${2:-} == worker ]]; then worker_logs "${3:-${workers[0]}}" -f --tail 100; else docker logs -f --tail 100 "${NAME}0"; fi ;;
  *) sed -n '5,12p' "$0" >&2; exit 2 ;;
esac
