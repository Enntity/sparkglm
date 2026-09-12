#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c source.sha256
nvcc --version
nvcc -std=c++17 -O2 -gencode=arch=compute_121a,code=sm_121a copy-test.cu copy.cu -o copy-test
sha256sum copy-test
