#!/usr/bin/env bash
# Build or serve a weightless, kernel-faithful GLM-5.3 miniature.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_CACHE_NAME="models--sparkglm--tinyglm"
MODEL_ROOT="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"
EXPERTS="${TINYGLM_EXPERTS:-16}"
MAX_LENGTH="${TINYGLM_MAX_MODEL_LEN:-32768}"

build() {
    python3 "$SCRIPT_DIR/scripts/make_tinyglm.py" \
        --output "$MODEL_ROOT" \
        --experts "$EXPERTS" \
        --max-length "$MAX_LENGTH"
}

case "${1:-build}" in
    build)
        build
        printf 'tinyGLM ready: %s (experts=%s max-length=%s)\n' \
            "$MODEL_ROOT" "$EXPERTS" "$MAX_LENGTH"
        ;;
    start|restart)
        command="$1"
        snapshot="$(build)"
        export MODEL="sparkglm/tinyglm"
        export MODEL_FALLBACK="$MODEL"
        export MODEL_CACHE_NAME
        export MODEL_FALLBACK_CACHE_NAME="$MODEL_CACHE_NAME"
        export MODEL_REVISION="${snapshot##*/}"
        export MODEL_FALLBACK_REVISION="$MODEL_REVISION"
        export EXPECTED_SHARDS=0
        export SKIP_DOWNLOAD=1
        export SPEC_METHOD=none
        export MTP_TOKENS=0
        export LANGUAGE_MODEL_ONLY=1
        export SERVED_MODEL_NAME="tinyGLM-5.3-EXL3"
        export MAX_MODEL_LEN="$MAX_LENGTH"
        export MAX_NUM_SEQS="${TINYGLM_MAX_NUM_SEQS:-4}"
        export MAX_NUM_BATCHED_TOKENS="${TINYGLM_MAX_NUM_BATCHED_TOKENS:-7168}"
        export GPU_MEM_UTIL="${TINYGLM_GPU_MEM_UTIL:-0.15}"
        export KV_CACHE_DTYPE="${TINYGLM_KV_CACHE_DTYPE:-fp8}"
        export SPARKGLM_TINY_DUMMY=1
        export GLM53_BOOT_SHAPE_WARMUP=0
        export GLM53_BOOT_LONG_C4=0
        export ABLIT=0
        export EXTRA_ARGS="--load-format dummy --generation-config vllm ${EXTRA_ARGS:-}"
        exec "$SCRIPT_DIR/start.sh" "$command"
        ;;
    status|logs|stop)
        exec "$SCRIPT_DIR/start.sh" "$@"
        ;;
    *)
        echo "usage: scripts/tinyglm.sh [build|start|restart|status|logs|stop]" >&2
        exit 2
        ;;
esac
