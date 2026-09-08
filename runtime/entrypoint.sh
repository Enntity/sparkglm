#!/usr/bin/env bash
# SPDX-License-Identifier: MIT AND Apache-2.0
# Adapted from LLooM's Mia launcher; SparkGLM supplies the built runtime.
set -euo pipefail

log() { printf '[sparkglm rank=%s] %s\n' "${NODE_RANK:-?}" "$*"; }

: "${NODE_RANK:?NODE_RANK is required}"
: "${CLUSTER_NODE_COUNT:?CLUSTER_NODE_COUNT is required}"
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MODEL_DIR:?MODEL_DIR is required}"

# The inherited DFlash loader preserves the target parallel configuration;
# accepting draft TP1 here would silently run TP2 and mislabel measurements.
if [[ "${SPEC_METHOD:-dflash}" == "dflash" && "${DFLASH_DRAFT_TP:-2}" != "${CLUSTER_NODE_COUNT}" ]]; then
  log "independent DFlash draft TP is not implemented by this adapter; use target TP=${CLUSTER_NODE_COUNT}"
  exit 1
fi

[[ -f "${MODEL_DIR}/config.json" ]] || {
  log "missing target config: ${MODEL_DIR}/config.json"
  exit 1
}

if [[ "${SPARKGLM_EXL3_E3:-0}" == "1" ]]; then
  [[ -f /usr/local/lib/python3.12/dist-packages/sparkglm_e3.py && -f /usr/local/lib/python3.12/dist-packages/exl3_fat_moe_ext.so ]] || {
    log "selected image does not contain the E3 adapter and extension"; exit 1;
  }
  if [[ "${SPARKGLM_EXL3_E3_POLICY:-large}" == "concurrent" ]]; then
    [[ -f /usr/local/lib/python3.12/dist-packages/sparkglm_e3_policy.py ]] || {
      log "selected image does not contain the concurrent E3 policy"; exit 1;
    }
  fi
fi
if [[ "${SPARKGLM_NVFP4_TINY:-0}" == "1" ]]; then
  [[ -f /usr/local/lib/python3.12/dist-packages/sparkglm_nvfp4_tiny.py && -f /usr/local/lib/python3.12/dist-packages/sparkglm_nvfp4_tiny.pth ]] || {
    log "selected image does not contain the guarded NVFP4 fixture initializer"; exit 1;
  }
fi
if [[ "${SPARKGLM_MXFP8_DRAFT:-0}" == "1" ]]; then
  grep -q '_fused_kv_weight_scale' /opt/glm53/patch_dflash2.py || {
    log "selected image does not contain MXFP8 DFlash2 context projection support"; exit 1;
  }
fi

if [[ "${SPEC_METHOD:-dflash}" == "dflash" && ! -f "${DFLASH_MODEL_DIR:-}/config.json" ]]; then
  log "missing DFlash2 config: ${DFLASH_MODEL_DIR:-unset}/config.json"
  exit 1
fi

# Runtime patches come from the selected SparkGLM image, never the legacy
# Mia files installed alongside LLooM. Image identity is pinned by the recipe.
for patch in \
  patch_glm_video_placeholders.py \
  patch_suppress_stops_in_reasoning.py \
  patch_scheduler_decode_floor.py \
  patch_glm5_drafter_group.py \
  patch_hybrid_prefix_hit.py \
  patch_xgrammar_termination.py \
  patch_kpool_tail_slotmap.py \
  patch_spinwait.py \
  patch_indexer_workspace.py \
  patch_ablit.py; do
  [[ -f "/opt/glm53/${patch}" ]] || {
    log "missing SparkGLM runtime patch: /opt/glm53/${patch}"
    exit 1
  }
  # These source patchers use only the standard library. Avoid importing the
  # serving stack through site .pth hooks ten times during each cold start.
  python3 -S "/opt/glm53/${patch}"
done

if [[ -z "${LIMIT_MM_PER_PROMPT:-}" ]]; then
  LIMIT_MM_PER_PROMPT='{"image":4,"video":1}'
fi

args=(
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash-exl3}"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "${VLLM_PORT:-8890}"
  --tensor-parallel-size "${CLUSTER_NODE_COUNT}"
  --nnodes "${CLUSTER_NODE_COUNT}"
  --node-rank "${NODE_RANK}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT:-29521}"
  --distributed-executor-backend mp
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --reasoning-parser glm45
  --enable-prefix-caching
  --no-enable-flashinfer-autotune
  --quantization "${QUANTIZATION:-exl3}"
  --max-model-len "${MAX_MODEL_LEN:-1000000}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.87}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-7168}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --chat-template /opt/glm53/chat_template.jinja
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
  --skip-mm-profiling
)

if [[ "${SPARKGLM_TINY_DUMMY:-0}" == "1" ]]; then
  python3 -S - "${MODEL_DIR}/config.json" <<'PYSAFE'
import json, sys
c = json.load(open(sys.argv[1]))
if c.get("quantization_config", {}).get("version") != "tinyglm-v1":
    raise SystemExit("dummy loading requires the synthetic tinyGLM fixture")
PYSAFE
  [[ "${SPEC_METHOD:-dflash}" == "none" ]] || { log "tinyGLM requires SPEC_METHOD=none"; exit 1; }
  args+=(--load-format dummy --generation-config vllm)
fi
if [[ "${SPARKGLM_NVFP4_TINY:-0}" == "1" ]]; then
  python3 -S - "${MODEL_DIR}/config.json" <<'PYSAFE'
import json, sys
if json.load(open(sys.argv[1])).get("_sparkglm_fixture") != "tinyglm-nvfp4-v1":
    raise SystemExit("NVFP4 dummy loading requires the synthetic fixture")
PYSAFE
  [[ "${SPEC_METHOD:-dflash}" == "none" && "${QUANTIZATION:-exl3}" == "compressed-tensors" ]] || { log "invalid NVFP4 fixture options"; exit 1; }
  args+=(--load-format dummy --generation-config vllm)
fi
if [[ "${LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
  args+=(--language-model-only)
fi

if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
  args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi
if [[ -n "${MOE_BACKEND:-}" ]]; then
  args+=(--moe-backend "${MOE_BACKEND}")
fi

if [[ "${NODE_RANK}" != "0" ]]; then
  args+=(--headless)
fi

case "${SPEC_METHOD:-dflash}" in
  dflash)
    dflash_tokens="${DFLASH_TOKENS:-7}"
    dflash_draft_tp="${DFLASH_DRAFT_TP:-2}"
    [[ "${dflash_tokens}" =~ ^[0-9]+$ ]] || { log "invalid DFLASH_TOKENS=${dflash_tokens}"; exit 1; }
    [[ "${dflash_draft_tp}" =~ ^[0-9]+$ ]] || { log "invalid DFLASH_DRAFT_TP=${dflash_draft_tp}"; exit 1; }
    # Do not launch Python here: Mia's installed video .pth emits a status line
    # on interpreter startup, which would contaminate command-substitution JSON.
    printf -v spec '{"method":"dflash","model":"%s","num_speculative_tokens":%d,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_tensor_parallel_size":%d}' \
      "${DFLASH_MODEL_DIR}" "${dflash_tokens}" "${dflash_draft_tp}"
    args+=(--speculative-config "${spec}")
    ;;
  mtp)
    args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-2}}")
    ;;
  none) ;;
  *)
    log "unsupported SPEC_METHOD=${SPEC_METHOD}"
    exit 1
    ;;
esac

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  args+=(--enforce-eager)
else
  args+=(--cudagraph-capture-sizes 1 2 4 8 16 24 32)
fi

log "starting SparkGLM quant=${QUANTIZATION:-exl3} TP=${CLUSTER_NODE_COUNT}, spec=${SPEC_METHOD:-dflash}, draft-tp=${DFLASH_DRAFT_TP:-2}, mnbt=${MAX_NUM_BATCHED_TOKENS:-7168}, E2=${EXL3_FAT_KERNEL:-1}"
exec vllm serve "${MODEL_DIR}" "${args[@]}"
