#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
sha256sum -c production.sha256
sha256sum bench bench.cu fixture.hpp gpu_fixture.cuh oracle.hpp
exec ./bench
