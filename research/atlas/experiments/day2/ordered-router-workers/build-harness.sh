#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Parent selects only a source-reviewed header/symbol and a fresh output filename.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
test "$#" -eq 3
header="$1"; kernel="$2"; output="$3"
[[ "$header" =~ ^[a-zA-Z0-9_.-]+\.cuh$ && "$output" =~ ^[a-zA-Z0-9_.-]+$ ]]
case "$kernel" in
  ordered_bf16_kernel|'obm::gemm3x288_kernel<true>'|dense_gemm_bf16_router) ;;
  *) exit 2 ;;
esac
test -f "$header"
test ! -e "$output"
printf '%s  %s\n' 53b1d1ae157124440ab7d6b8a7bc2d63aa2c1d1ff2e45f713bf4e4ec4f33c5af production/dense_gemm_bf16.cu | sha256sum -c -
nvcc --version
exec nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a \
  --fmad=false --ftz=false --prec-div=true -Xcompiler=-ffp-contract=off \
  -Xptxas=-v "-DCANDIDATE_HEADER=\"$header\"" "-DCANDIDATE_KERNEL=$kernel" bench.cu -o "$output"
