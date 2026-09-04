#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Rebuild the two native components changed by the recorded video foundation.
set -euo pipefail

source_dir="/opt/video-vllm-source"
build_dir="/opt/video-vllm-build"
install_dir="/opt/video-native"
revision="487ecf187d3dfe74d2cf6119a92881dba403c219"
mkdir -p "$source_dir" "$build_dir" "$install_dir"
curl --fail --location --retry 3 \
    "https://github.com/vllm-project/vllm/archive/${revision}.tar.gz" \
    | tar -xz -C "$source_dir" --strip-components=1
git -C "$source_dir" apply --check /opt/video-native.patch
git -C "$source_dir" apply /opt/video-native.patch

# Only these native targets changed. Avoid configuring unrelated optional
# packages; preserve their already installed binaries from the pinned base.
python3 - "$source_dir" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1]) / "CMakeLists.txt"
text = path.read_text()
anchor = "# For CUDA and HIP builds also build the triton_kernels external package."
if text.count(anchor) != 1:
    raise RuntimeError("native-only configure anchor drifted")
text = text.replace(anchor,
    'include(cmake/external_projects/deepgemm.cmake)\nreturn()\n\n' + anchor)
path.write_text(text)
PY

export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=12.1a
export CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$CPATH"
export C_INCLUDE_PATH="$CPATH"
# The runtime base ships versioned NVRTC without the development linker name.
ln -sf libnvrtc.so.13 /usr/local/cuda/targets/sbsa-linux/lib/libnvrtc.so
# Generator outputs live in the fresh source tree, while CMake's "already
# generated" markers live in the persistent build cache. Recreate those
# outputs even when the generator script itself has not changed.
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -U '*_GEN_SCRIPT_HASH_AND_ARCH' \
    -DCMAKE_BUILD_TYPE=Release \
    -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDA_nvrtc_LIBRARY=/usr/local/cuda/targets/sbsa-linux/lib/libnvrtc.so.13 \
    -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DFETCHCONTENT_BASE_DIR=/opt/video-deps \
    -DNVCC_THREADS=1
cmake --build "$build_dir" --parallel "${MAX_JOBS:-8}" \
    --target _C_stable_libtorch _deep_gemm_C
cmake --install "$build_dir" --component _C_stable_libtorch
cmake --install "$build_dir" --component _deep_gemm_C
cp "$source_dir/LICENSE" "$install_dir/VLLM-LICENSE"
