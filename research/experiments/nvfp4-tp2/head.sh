#!/bin/bash
set -euo pipefail
say() { echo "[glm53-exl3-head] $*"; }

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${API_HOST:-127.0.0.1}"
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 0
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
)
[ -n "${DCP:-}" ] && ARGS+=(--decode-context-parallel-size "${DCP}")
[ -n "${CP_KV_CACHE_INTERLEAVE_SIZE:-}" ] && ARGS+=(--cp-kv-cache-interleave-size "${CP_KV_CACHE_INTERLEAVE_SIZE}")
[ -n "${DCP_COMM_BACKEND:-}" ] && ARGS+=(--dcp-comm-backend "${DCP_COMM_BACKEND}")
[ -n "${ATTENTION_BACKEND:-}" ] && ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
[ -n "${MOE_BACKEND:-}" ] && ARGS+=(--moe-backend "${MOE_BACKEND}")
[ -n "${LINEAR_BACKEND:-}" ] && ARGS+=(--linear-backend "${LINEAR_BACKEND}")
[ -n "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}" ] && ARGS+=(--max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}")
[ -n "${CUDAGRAPH_MODE:-}" ] && ARGS+=(--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"custom_ops\":[\"all\"]}")
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
if [ "${SPEC_METHOD:-mtp}" = "dflash" ]; then
    ARGS+=(--speculative-config "$(python3 -c 'import json,os
spec={"method":"dflash","model":os.environ["DFLASH_MODEL_DIR"],"num_speculative_tokens":int(os.environ.get("DFLASH_TOKENS","7")),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}
tp=os.environ.get("DFLASH_DRAFT_TP","").strip()
if tp:
    spec["draft_tensor_parallel_size"]=int(tp)
print(json.dumps(spec,separators=(",",":")))')")
elif [ "${SPEC_METHOD:-mtp}" = "none" ]; then
    :
elif [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "$(python3 -c 'import json,os
spec={"method":"mtp","num_speculative_tokens":int(os.environ["MTP_TOKENS"])}
if os.environ.get("MTP_MOE_BACKEND","").strip(): spec["moe_backend"]=os.environ["MTP_MOE_BACKEND"]
if os.environ.get("MTP_ATTENTION_BACKEND","").strip(): spec["attention_backend"]=os.environ["MTP_ATTENTION_BACKEND"]
print(json.dumps(spec,separators=(",",":")))')")
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} skip-mm-profiling=${SKIP_MM_PROFILING:-1} chat-template=${CHAT_TEMPLATE:-}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
if [ -f /opt/glm53/patch_glm_video_placeholders.py ]; then
    python3 /opt/glm53/patch_glm_video_placeholders.py
fi
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
