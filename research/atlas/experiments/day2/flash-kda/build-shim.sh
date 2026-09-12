#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd /work
nvcc -shared -Xcompiler -fPIC -O3 --fmad=false \
  -gencode=arch=compute_121a,code=sm_121a runtime_bridge.cu \
  -L/work -latlas_glm53_flash_kda -Xlinker -rpath -Xlinker '$ORIGIN' \
  -o /work/libatlas_mango_flash.so
g++ -x c++ native_smoke.cu -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 -lcudart -L/work -latlas_mango_flash \
  -Wl,-rpath,'$ORIGIN' -o /work/native_smoke
