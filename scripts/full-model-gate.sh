#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Collect the frozen full-model G3 matrix for one baseline or candidate arm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

arm="${1:?usage: scripts/full-model-gate.sh baseline|candidate OUTPUT_DIR MODEL}"
output_dir="${2:?usage: scripts/full-model-gate.sh baseline|candidate OUTPUT_DIR MODEL}"
model="${3:?usage: scripts/full-model-gate.sh baseline|candidate OUTPUT_DIR MODEL}"
case "$arm" in baseline|candidate) ;; *) echo "arm must be baseline or candidate" >&2; exit 2 ;; esac

base_url="${SPARKGLM_BASE_URL:-http://127.0.0.1:8888}"
pair_id="${SPARKGLM_PAIR_ID:?set SPARKGLM_PAIR_ID to the same stable value for both arms}"
repetitions="${SPARKGLM_FULL_REPETITIONS:-3}"
only_repetition="${SPARKGLM_FULL_REPETITION_ID:-}"
stagger_ms="${SPARKGLM_FULL_STAGGER_MS:-1000}"
mkdir -p "$output_dir/raw"

if [ -z "${SPARKGLM_IMAGE_DIGESTS:-}" ] || [ -z "${SPARKGLM_MODEL_REVISION:-}" ]; then
    echo "set SPARKGLM_IMAGE_DIGESTS to both rank digests and SPARKGLM_MODEL_REVISION to the immutable model revision" >&2
    exit 2
fi
python3 scripts/capture_server_manifest.py \
    --base-url "$base_url" \
    --model "$model" \
    --api-key "${SPARKGLM_API_KEY:-}" \
    --output "$output_dir/raw/${arm}-server-manifest.json"

run_case() {
    local name="$1" concurrency="$2" prompt_tokens="$3" repetition="$4" output="$5"
    local -a auth_args=()
    [ -n "${SPARKGLM_API_KEY:-}" ] && auth_args=(--api-key "$SPARKGLM_API_KEY")
    python3 benchmarks/staggered_openai.py \
        --base-url "$base_url" \
        --model "$model" \
        --concurrency "$concurrency" \
        --stagger-ms "$stagger_ms" \
        --prompt-tokens "$prompt_tokens" \
        --prompt-salt "${pair_id}-${name}-r${repetition}" \
        --output-tokens 400 \
        --min-output-tokens 400 \
        --exact-prompt-tokens \
        "${auth_args[@]}" \
        --timeout-s "${SPARKGLM_FULL_TIMEOUT_S:-900}" \
        > "$output"
    python3 - "$output" "$concurrency" "$prompt_tokens" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
expected_concurrency = int(sys.argv[2])
expected_prompt_tokens = int(sys.argv[3])
assert payload["summary"]["successful"] == expected_concurrency
assert payload["config"]["exact_prompt_tokens"] is True
assert len(payload["requests"]) == expected_concurrency
for request in payload["requests"]:
    assert request["error"] is None
    assert request["prompt_tokens"] == expected_prompt_tokens, request
    assert request["completion_tokens"] == 400, request
    assert request["reached_requested_completion_tokens"] is True, request
    assert request["output_chars"] > 0, request
    assert request["own_request_marker"] is True, request
    assert request["foreign_request_markers"] == [], request
PY
}

if [ "${SPARKGLM_SKIP_DISCARDED_WARMUP:-0}" != "1" ]; then
    echo "==> discarded exact 16K C4 workload warmup"
    run_case warmup-c4-16k 4 16384 0 "$output_dir/.discarded-warmup.json"
fi

if [ -n "$only_repetition" ]; then
    case "$only_repetition" in
        *[!0-9]*|"") echo "SPARKGLM_FULL_REPETITION_ID must be a positive integer" >&2; exit 2 ;;
    esac
    [ "$only_repetition" -ge 1 ] || { echo "SPARKGLM_FULL_REPETITION_ID must be positive" >&2; exit 2; }
    repetition_ids="$only_repetition"
else
    repetition_ids="$(seq 1 "$repetitions")"
fi

for repetition in $repetition_ids; do
    echo "==> $arm repetition $repetition/$repetitions"
    run_case c1-16k 1 16384 "$repetition" "$output_dir/raw/${arm}-c1-16k-r${repetition}.json"
    run_case c1-32k 1 32768 "$repetition" "$output_dir/raw/${arm}-c1-32k-r${repetition}.json"
    run_case c2-16k 2 16384 "$repetition" "$output_dir/raw/${arm}-c2-16k-r${repetition}.json"
    run_case c2-32k 2 32768 "$repetition" "$output_dir/raw/${arm}-c2-32k-r${repetition}.json"
    run_case c4-16k 4 16384 "$repetition" "$output_dir/raw/${arm}-c4-16k-r${repetition}.json"
done

rm -f "$output_dir/.discarded-warmup.json"
echo "G3 $arm evidence written to $output_dir/raw"
