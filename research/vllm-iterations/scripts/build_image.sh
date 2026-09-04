#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
cd "${repo_root}"

if [[ $(uname -m) != "aarch64" ]]; then
  echo "SparkGLM images must be built on an aarch64 CUDA host." >&2
  exit 2
fi

source_revision=$(git rev-parse HEAD)
image_tag=${1:-sparkglm-vllm:${source_revision:0:8}}
build_jobs=${SPARKGLM_BUILD_JOBS:-2}
nvcc_threads=${SPARKGLM_NVCC_THREADS:-1}
vllm_base=${SPARKGLM_VLLM_BASE:-487ecf187d3dfe74d2cf6119a92881dba403c219}
use_precompiled=${SPARKGLM_USE_PRECOMPILED:-0}

case ${build_jobs} in
  ''|*[!0-9]*|0) echo "SPARKGLM_BUILD_JOBS must be a positive integer." >&2; exit 2 ;;
esac
case ${nvcc_threads} in
  ''|*[!0-9]*|0) echo "SPARKGLM_NVCC_THREADS must be a positive integer." >&2; exit 2 ;;
esac
if [[ ${use_precompiled} != 0 && ${use_precompiled} != 1 ]]; then
  echo "SPARKGLM_USE_PRECOMPILED must be 0 or 1." >&2
  exit 2
fi

precompiled_args=()
if [[ ${use_precompiled} == 1 ]]; then
  precompiled_args+=(
    --build-arg VLLM_USE_PRECOMPILED=1
    --build-arg "VLLM_MERGE_BASE_COMMIT=${vllm_base}"
    --build-arg VLLM_MAIN_CUDA_VERSION=13.0
  )
fi

echo "Building ${image_tag} from ${source_revision} (jobs=${build_jobs}, nvcc=${nvcc_threads}, precompiled=${use_precompiled})"
docker buildx build --load \
  --platform linux/arm64 \
  --target sparkglm-openai \
  --build-arg BUILD_BASE_IMAGE=pytorch/manylinuxaarch64-builder:cuda13.0 \
  "${precompiled_args[@]}" \
  --build-arg max_jobs="${build_jobs}" \
  --build-arg nvcc_threads="${nvcc_threads}" \
  --build-arg vllm_build_flash_attn=ON \
  --build-arg vllm_build_flash_attn3=OFF \
  --build-arg torch_cuda_arch_list=12.1a \
  --build-arg SPARKGLM_SOURCE_REVISION="${source_revision}" \
  -t "${image_tag}" \
  -f docker/Dockerfile \
  .

docker image inspect "${image_tag}" \
  --format 'image={{.Id}} bytes={{.Size}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
