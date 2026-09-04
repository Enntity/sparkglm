#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    echo "Usage: $0 BASE_IMAGE OUTPUT_IMAGE SOURCE_REVISION VLLM_PYTHON_FILE [...]" >&2
    echo "Set SPARKGLM_OVERLAY_WORKER=user@host to build the same sparse overlay there." >&2
    exit 2
}

[[ $# -ge 4 ]] || usage

base_image=$1
output_image=$2
source_revision=$3
shift 3

case "${source_revision}" in
    *[!A-Za-z0-9._-]*)
        echo "SOURCE_REVISION must be a label-safe revision name" >&2
        exit 2
        ;;
esac

repo_root=$(git rev-parse --show-toplevel)
context_dir=$(mktemp -d "${TMPDIR:-/tmp}/sparkglm-overlay.XXXXXX")
trap 'rm -rf "${context_dir}"' EXIT

for source_file in "$@"; do
    case "${source_file}" in
        vllm/*.py) ;;
        *)
            echo "Overlay inputs must be Python files below vllm/: ${source_file}" >&2
            exit 2
            ;;
    esac
    [[ -f "${repo_root}/${source_file}" ]] || {
        echo "Missing overlay input: ${source_file}" >&2
        exit 2
    }
    destination="${context_dir}/${source_file}"
    mkdir -p "$(dirname "${destination}")"
    cp "${repo_root}/${source_file}" "${destination}"
done

docker build \
    --build-arg "BASE_IMAGE=${base_image}" \
    --build-arg "SPARKGLM_SOURCE_REVISION=${source_revision}" \
    --file "${repo_root}/docker/Dockerfile.sparkglm-python-overlay" \
    --tag "${output_image}" \
    "${context_dir}"

if [[ -n "${SPARKGLM_OVERLAY_WORKER:-}" ]]; then
    remote_context="/tmp/sparkglm-overlay-${source_revision}"
    ssh "${SPARKGLM_OVERLAY_WORKER}" mkdir -p "${remote_context}"
    scp -r "${context_dir}/." "${SPARKGLM_OVERLAY_WORKER}:${remote_context}/"
    scp "${repo_root}/docker/Dockerfile.sparkglm-python-overlay" \
        "${SPARKGLM_OVERLAY_WORKER}:${remote_context}/Dockerfile"
    ssh "${SPARKGLM_OVERLAY_WORKER}" docker build \
        --build-arg "BASE_IMAGE=${base_image}" \
        --build-arg "SPARKGLM_SOURCE_REVISION=${source_revision}" \
        --file "${remote_context}/Dockerfile" \
        --tag "${output_image}" \
        "${remote_context}"
fi
