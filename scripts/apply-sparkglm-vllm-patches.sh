#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
VLLM_SOURCE="${1:?usage: $0 /path/to/vllm-source}"
VLLM_BASE="487ecf187d3dfe74d2cf6119a92881dba403c219"

git -C "$VLLM_SOURCE" rev-parse --git-dir >/dev/null 2>&1 || {
    echo "not a vLLM git checkout: $VLLM_SOURCE" >&2
    exit 2
}

actual="$(git -C "$VLLM_SOURCE" rev-parse HEAD)"
[ "$actual" = "$VLLM_BASE" ] || {
    echo "vLLM source must be exactly $VLLM_BASE (got $actual)" >&2
    exit 2
}

[ -z "$(git -C "$VLLM_SOURCE" status --porcelain)" ] || {
    echo "vLLM source checkout is dirty; refusing to blend patches" >&2
    exit 2
}

for patch in \
    "$REPO_ROOT/patches/vllm/packed-rmsnorm.patch" \
    "$REPO_ROOT/patches/vllm/native-fp16-sparse-selector.patch"; do
    git -C "$VLLM_SOURCE" apply --check "$patch"
    git -C "$VLLM_SOURCE" apply "$patch"
done

echo "SparkGLM vLLM patches applied to $VLLM_SOURCE"
