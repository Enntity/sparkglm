#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# SparkGLM on four Sparks. Run from any machine with ssh to all four: it reads
# their ports, finds whether they are cabled as a ring or a chain, and starts one
# rank on each, workers first. The first host is rank 0 and serves the API on its
# own 127.0.0.1:8893; a chain must start at one of its ends.
#
#   install/start-four.sh fabric|up|stop|status HOST0 HOST1 HOST2 HOST3
#
# Set IMAGE and MODEL_ROOT (the same absolute path on every host) for up. Optional:
# MODE (spark), STRICT_MTP (0), TOPOLOGY (chain, to leave a ring's closing link
# unused), PROFILE (4x1m-tp4), KV_SHARD (1), GPU_MEMORY_UTILIZATION, PREFIX_CACHE_DIR,
# PREFIX_CACHE_GB, TOOL_GUARD (on).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
usage() { sed -n '8p' "$0" | cut -c3- >&2; exit 2; }
[[ $# == 5 ]] || usage
action=$1; shift; hosts=("$@")
NAME=atlas-sparkglm-rank
API=http://127.0.0.1:8893
on() { ssh -n -o BatchMode=yes "$1" "${@:2}"; }

detect() {
  local inventories=() host
  for host in "${hosts[@]}"; do
    inventories+=("$(ssh -o BatchMode=yes "$host" bash -s < fabric-ports.sh)")
  done
  python3 ../fabric.py topology "${inventories[@]}"
}

stop() {
  local host
  for host in "${hosts[@]}"; do
    on "$host" "docker rm -f ${NAME}0 ${NAME}1 ${NAME}2 ${NAME}3 >/dev/null 2>&1 || true"
  done
}

up() {
  : "${IMAGE:?set IMAGE}" "${MODEL_ROOT:?set MODEL_ROOT to the same absolute path on every host}"
  local plan rank host kind iface leader left right mode=${MODE:-spark} common
  plan=$(detect)
  common=(--world-size 4 --model-root "$MODEL_ROOT" --image "$IMAGE" --profile "${PROFILE:-4x1m-tp4}" --mode "$mode"
          --tool-guard "${TOOL_GUARD:-on}"
          ${GPU_MEMORY_UTILIZATION:+--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"}
          ${PREFIX_CACHE_DIR:+--prefix-cache-dir "$PREFIX_CACHE_DIR"} ${PREFIX_CACHE_GB:+--prefix-cache-gb "$PREFIX_CACHE_GB"})
  [[ ${KV_SHARD:-1} == 1 ]] && common+=(--kv-shard)
  [[ ${STRICT_MTP:-0} == 1 ]] && common+=(--strict-mtp)
  stop
  while read -r rank host kind iface leader left right; do
    if [[ ${TOPOLOGY:-$kind} == chain ]]; then
      kind=chain; ((rank == 0)) && left=-; ((rank == 3)) && right=-
    fi
    echo "rank $rank on ${hosts[$host]}: $kind, $iface, left ${left}, right ${right}"
    on "${hosts[$host]}" ping -c1 -W2 "$leader" >/dev/null ||
      { echo "${hosts[$host]} cannot reach the leader at $leader" >&2; exit 1; }
    args=(--rank "$rank" --topology "$kind" --leader-address "$leader" --fabric-interface "$iface" "${common[@]}")
    [[ $left != - ]] && args+=(--left-hca "$left")
    [[ $right != - ]] && args+=(--right-hca "$right")
    ssh -o BatchMode=yes "${hosts[$host]}" "bash -s -- $(printf '%q ' "${args[@]}")" < start-node.sh
  done < <(sort -rn <<< "$plan")
  for ((i = 0; i < 120; i++)); do
    if on "${hosts[0]}" curl -sf "$API/health" >/dev/null; then
      on "${hosts[0]}" curl -fsS --max-time 300 "$API/v1/chat/completions" -H "'Content-Type: application/json'" \
        -d "'{\"model\": \"glm-5.3-flash-atlas\", \"max_tokens\": 1024, \"temperature\": 0, \"top_p\": 1, \"reasoning_effort\": \"low\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 17*23? Answer with the number.\"}]}'" |
        python3 smoke.py
      echo "ready: ${hosts[0]} $API/v1 model glm-5.3-flash-atlas ($mode)"
      return
    fi
    while read -r rank host _; do
      [[ $(on "${hosts[$host]}" docker inspect -f '{{.State.Status}}' "$NAME$rank" 2>/dev/null) == running ]] ||
        { on "${hosts[$host]}" docker logs --tail 30 "$NAME$rank" 2>&1; echo "rank $rank stopped" >&2; exit 1; }
    done <<< "$plan"
    sleep 10
  done
  echo 'not ready after 20 minutes' >&2; exit 1
}

case $action in
  fabric) detect ;;
  up) up ;;
  stop) stop ;;
  status) on "${hosts[0]}" curl -sf "$API/health" >/dev/null && echo "health ok on ${hosts[0]}" || echo 'health: not serving' ;;
  *) usage ;;
esac
