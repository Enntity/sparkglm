#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the Atlas native ABI from source, including the formerly cached object.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: build-native.sh ENGINE_SOURCE FLASHINFER_SOURCE PRODUCT_SOURCE OUTPUT' >&2
  exit 2
fi
engine=$(realpath "$1")
flashinfer=$(realpath "$2")
product=$(realpath "$3")
mkdir -p "$4"
output=$(realpath "$4")
cuda=${CUDA_HOME:-/usr/local/cuda}
revision=8eccd0c1352165302840c0e19066bc42d36dbd7a
[[ $(git -c "safe.directory=$flashinfer" -C "$flashinfer" rev-parse HEAD) == "$revision" ]]
[[ -z $(git -c "safe.directory=$flashinfer" -C "$flashinfer" status --porcelain --untracked-files=all) ]]
wrapper="$engine/scripts/dev/glm_sparse_native"
# The ABI shims declare the pinned upstream enum. Fail if either definition drifts.
cmp "$wrapper/reference/traits/model/model_type.h" \
    "$flashinfer/include/flashinfer/attention/sparse_mla_sm120/model/model_type.h"
mkdir -p "$output/build" "$output/notices"
flags=(-std=c++17 -O2 -gencode=arch=compute_121a,code=sm_121a -Xcompiler=-fPIC)
# CUDA 13 otherwise makes template host stubs local. The preallocation init shim
# must resolve this exact GLM kernel without launching or allocating GPU memory.
# Match FlashInfer jit/core.py's arithmetic and compilation flags. The prep
# translation unit below deliberately keeps its separately qualified precise flags.
"$cuda/bin/nvcc" "${flags[@]}" -O3 --use_fast_math -Xfatbin=-compress-all \
  -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 \
  -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 \
  --static-global-template-stub=false -I"$flashinfer/include" \
  -c "$flashinfer/csrc/sparse_mla_sm120_prefill.cu" -o "$output/build/sparse-prefill.o"
"$cuda/bin/nvcc" "${flags[@]}" --fmad=false --prec-div=true --ftz=false \
  -c "$wrapper/atlas-glm-sparse-native.cu" -o "$output/build/sparse-prep.o"
g++ -std=c++17 -O2 -fPIC -Wall -Wextra -Werror -I"$cuda/include" -I"$wrapper" \
  -c "$wrapper/native-init.cpp" -o "$output/build/sparse-init.o"
g++ -std=c++17 -O2 -fPIC -shared -Wall -Wextra -I"$cuda/include" -I"$wrapper" \
  "$wrapper/native-bridge.cpp" "$output/build/sparse-init.o" \
  "$output/build/sparse-prep.o" "$output/build/sparse-prefill.o" \
  -L"$cuda/lib64" -Wl,--no-undefined -Wl,-z,defs -lcudart \
  -o "$output/libatlas_glm_sparse_native.so"
if readelf -d "$output/libatlas_glm_sparse_native.so" | grep -Eiq 'NEEDED.*(tvm|torch|python)'; then
  echo 'Unexpected framework dependency in native sparse library' >&2
  exit 1
fi
cp "$flashinfer/LICENSE" "$output/notices/FlashInfer-LICENSE"
cp "$flashinfer/csrc/sparse_mla_sm120_prefill.cu" "$output/notices/NVIDIA-sparse-prefill-source.cu"
if [[ ${ATLAS_BUILD_SPARSE_ONLY:-0} != 1 ]]; then
  FLASH_KDA_OUTPUT="$output" NVCC_THREADS="${NVCC_THREADS:-2}" \
    bash "$product/research/atlas/flash_kda/rebuild.sh"
  "$cuda/bin/nvcc" -std=c++17 -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a \
    -shared -Xcompiler=-fPIC "$engine/research/flash-kda-prefill/runtime_bridge.cu" \
    -L"$output" -latlas_glm53_flash_kda -Xlinker -rpath -Xlinker '$ORIGIN' \
    -o "$output/libatlas_mango_flash.so"
  cp "$product/research/atlas/flash_kda/LICENSE" "$output/notices/FlashKDA-LICENSE"
fi
(cd "$output" && sha256sum ./*.so > native-libraries.sha256)
