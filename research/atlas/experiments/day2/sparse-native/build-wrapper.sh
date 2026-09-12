#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -gencode=arch=compute_121a,code=sm_121a -I/source /eval/atlas-wrapper-reviewed.cu -o /eval/libatlas_sparse.so
sha256sum /eval/libatlas_sparse.so
