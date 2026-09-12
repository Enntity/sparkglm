#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c production.sha256
nvcc --version
exec nvcc -std=c++17 -O3 -gencode=arch=compute_121a,code=sm_121a \
  --fmad=false -Xcompiler=-ffp-contract=off -Xptxas=-v bench.cu -o bench
