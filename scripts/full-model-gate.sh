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
stagger_ms="${SPARKGLM_FULL_STAGGER_MS:-1000}"
mkdir -p "$output_dir/raw"

run_case() {
    local name="$1" concurrency="$2" prompt_tokens="$3" repetition="$4" output="$5"
    python3 benchmarks/staggered_openai.py \
        --base-url "$base_url" \
        --model "$model" \
        --concurrency "$concurrency" \
        --stagger-ms "$stagger_ms" \
        --prompt-tokens "$prompt_tokens" \
        --prompt-salt "${pair_id}-${name}-r${repetition}" \
        --output-tokens 400 \
        --min-output-tokens 400 \
        --timeout-s "${SPARKGLM_FULL_TIMEOUT_S:-900}" \
        > "$output"
    python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); expected=int(sys.argv[2]); assert p["summary"]["successful"] == expected' \
        "$output" "$concurrency"
}

if [ "${SPARKGLM_SKIP_DISCARDED_WARMUP:-0}" != "1" ]; then
    echo "==> discarded exact 16K C4 workload warmup"
    run_case warmup-c4-16k 4 16384 0 "$output_dir/.discarded-warmup.json"
fi

for repetition in $(seq 1 "$repetitions"); do
    echo "==> $arm repetition $repetition/$repetitions"
    run_case c1-16k 1 16384 "$repetition" "$output_dir/raw/${arm}-c1-16k-r${repetition}.json"
    run_case c1-32k 1 32768 "$repetition" "$output_dir/raw/${arm}-c1-32k-r${repetition}.json"
    run_case c2-16k 2 16384 "$repetition" "$output_dir/raw/${arm}-c2-16k-r${repetition}.json"
    run_case c2-32k 2 32768 "$repetition" "$output_dir/raw/${arm}-c2-32k-r${repetition}.json"
    run_case c4-16k 4 16384 "$repetition" "$output_dir/raw/${arm}-c4-16k-r${repetition}.json"
done

rm -f "$output_dir/.discarded-warmup.json"
echo "G3 $arm evidence written to $output_dir/raw"
