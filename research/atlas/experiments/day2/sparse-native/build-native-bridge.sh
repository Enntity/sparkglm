#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
object=${1:-/native/csrc_sparse_mla_sm120_prefill.cuda.o}
output=${2:-/tmp/libatlas_sparse_native.so}
cuda_root=${CUDA_HOME:-/usr/local/cuda}
test -f "$object"
# CPU-only compile/link. No nvcc, Python, Torch, TVM, JIT, network or GPU calls.
g++ -std=c++17 -O2 -fPIC -shared -Wall -Wextra \
  -I"$cuda_root/include" -I"$source_dir" "$source_dir/native-bridge.cpp" "$object" \
  -L"$cuda_root/lib64" -Wl,--no-undefined -Wl,-z,defs -lcudart -o "$output"
readelf -d "$output" | grep NEEDED
if readelf -d "$output" | grep -Eiq 'NEEDED.*(tvm|torch|python)'; then
  echo 'FAIL: unexpected framework dependency' >&2
  exit 1
fi
sha256sum "$object" "$source_dir/native-bridge.cpp" "$source_dir/native-bridge.h" \
  "$source_dir/reference/traits/model/model_type.h" "$output"
