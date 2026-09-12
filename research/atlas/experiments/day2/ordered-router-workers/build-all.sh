#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c bundle.sha256
bash build-harness.sh existing-router.cuh dense_gemm_bf16_router bench-existing
bash build-harness.sh ds-reviewed.cuh ordered_bf16_kernel bench-ds
bash build-harness.sh glm-reviewed.cuh 'obm::gemm3x288_kernel<true>' bench-glm
sha256sum bench-existing bench-ds bench-glm
