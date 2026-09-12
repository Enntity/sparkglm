#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd /eval
object=/native/csrc_sparse_mla_sm120_prefill.cuda.o
expected=9e372b5a47ade0a8332a73878ec3f6b917ae137447fea2335d2003dd146d549a
actual=$(sha256sum "$object" | cut -d' ' -f1)
test "$actual" = "$expected"
# Keep this compile separate from the prebuilt upstream translation unit.
# The reviewed prep scale deliberately matches Torch's scalar reciprocal path.
nvcc -std=c++17 -O2 --fmad=false --prec-div=true --ftz=false \
  -gencode=arch=compute_121a,code=sm_121a -Xcompiler=-fPIC \
  -c atlas-glm-sparse-native.cu -o /tmp/atlas-glm-sparse-native.o
g++ -std=c++17 -O2 -fPIC -Wall -Wextra -Werror \
  -I/usr/local/cuda/include -I/eval -c native-init.cpp -o /tmp/native-init.o
g++ -std=c++17 -O2 -fPIC -shared -Wall -Wextra \
  -I/usr/local/cuda/include -I/eval native-bridge.cpp /tmp/native-init.o \
  /tmp/atlas-glm-sparse-native.o "$object" -L/usr/local/cuda/lib64 \
  -Wl,--no-undefined -Wl,-z,defs -lcudart -o /eval/libatlas_glm_sparse_native.so
readelf -d /eval/libatlas_glm_sparse_native.so | grep NEEDED
if readelf -d /eval/libatlas_glm_sparse_native.so | grep -Eiq 'NEEDED.*(tvm|torch|python)'; then exit 1; fi
sha256sum /eval/libatlas_glm_sparse_native.so
