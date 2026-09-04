#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Exercise the real 16K staggered C4 shape and prove that four requests can be
# resident together. A short-shape warmup cannot detect an undersized hybrid
# KDA/MLA state pool.
set -u

BASE="${1:-http://127.0.0.1:8888}"
MODEL="${2:-GLM-5.3-Flash-EXL3}"
CURL_BIN="${WARMUP_CURL:-curl}"
REQ_TIMEOUT="${GLM53_WARMUP_REQ_TIMEOUT:-240}"
FACTS="${GLM53_C4_WARMUP_FACTS:-1333}"
NONCE="$$-$(date +%s)"

AUTH_ARGS=()
if [ -n "${GLM53_WARMUP_BEARER:-}" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer ${GLM53_WARMUP_BEARER}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
    AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

make_prompt() {
    local lane="$1" fact
    printf 'Capacity warmup %s lane %s. Read the unique reference context.\\n' \
        "$NONCE" "$lane"
    for ((fact = 0; fact < FACTS; fact++)); do
        printf 'context-%s-%s: alpha beta gamma delta\\n' "$lane" "$fact"
    done
    printf 'Reply with numbered concise systems facts.'
}

make_payload() {
    local lane="$1"
    {
        printf '{"model":"%s","messages":[{"role":"user","content":"' "$MODEL"
        make_prompt "$lane"
        printf '"}],"stream":true,"max_tokens":400,"temperature":0,'
        printf '"top_p":1,"ignore_eos":true,'
        printf '"chat_template_kwargs":{"enable_thinking":false}}'
    } >"$tmpdir/payload-$lane.json"
}

fire() {
    local lane="$1"
    sleep "$((lane - 1))"
    if "$CURL_BIN" -fsS -N --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
        "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
        --data-binary "@$tmpdir/payload-$lane.json" >/dev/null \
        2>>"$tmpdir/errors"; then
        printf 'ok\n' >"$tmpdir/result-$lane"
    else
        printf 'fail\n' >"$tmpdir/result-$lane"
    fi
}

for lane in 1 2 3 4; do
    make_payload "$lane"
done

tokenize_payload="$(make_prompt 1)"
tokenize_response=$("$CURL_BIN" -fsS --max-time 30 "${AUTH_ARGS[@]}" \
    "$BASE/tokenize" -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL"'","prompt":"'"$tokenize_payload"'"}' \
    2>>"$tmpdir/errors") || {
    echo "c4-capacity-warmup: tokenize failed" >&2
    exit 1
}
prompt_tokens=$(printf '%s\n' "$tokenize_response" \
    | grep -o '"count"[[:space:]]*:[[:space:]]*[0-9]*' \
    | head -n 1 | grep -o '[0-9]*$')
if [ -z "$prompt_tokens" ] || [ "$prompt_tokens" -lt 15000 ] \
    || [ "$prompt_tokens" -gt 17000 ]; then
    echo "c4-capacity-warmup: expected a 15K-17K prompt, got ${prompt_tokens:-unknown}" >&2
    exit 1
fi

pids=()
for lane in 1 2 3 4; do
    fire "$lane" &
    pids+=("$!")
done

started=$(date +%s)
max_running=0
max_waiting=0
max_kv=0
while :; do
    alive=0
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done
    metrics=$("$CURL_BIN" -fsS --max-time 5 "${AUTH_ARGS[@]}" "$BASE/metrics" \
        2>>"$tmpdir/errors" || true)
    running=$(printf '%s\n' "$metrics" \
        | awk '/^vllm:num_requests_running[{]/{print int($2); exit}')
    waiting=$(printf '%s\n' "$metrics" \
        | awk '/^vllm:num_requests_waiting[{]/{print int($2); exit}')
    kv=$(printf '%s\n' "$metrics" \
        | awk '/^vllm:kv_cache_usage_perc[{]/{print $2; exit}')
    running="${running:-0}"
    waiting="${waiting:-0}"
    kv="${kv:-0}"
    [ "$running" -gt "$max_running" ] && max_running="$running"
    [ "$waiting" -gt "$max_waiting" ] && max_waiting="$waiting"
    max_kv=$(awk -v old="$max_kv" -v new="$kv" \
        'BEGIN { print (new > old) ? new : old }')
    [ "$alive" -eq 0 ] && break
    sleep 1
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done
elapsed=$(($(date +%s) - started))

echo "c4-capacity-warmup: prompt=${prompt_tokens} tokens/stream max_running=${max_running} max_waiting=${max_waiting} peak_kv=${max_kv} wall=${elapsed}s"
if [ "$failed" -ne 0 ]; then
    echo "c4-capacity-warmup: ${failed}/4 requests failed" >&2
    sed -n '1,5p' "$tmpdir/errors" >&2 2>/dev/null || true
    exit 1
fi
if [ "$max_running" -lt 4 ]; then
    echo "c4-capacity-warmup: C4 admission FAILED; configured max-num-seqs=4 but only ${max_running} requests became resident" >&2
    exit 1
fi
