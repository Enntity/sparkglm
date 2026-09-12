#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Pinned NVIDIA NVFP4 + DFlash2 package; same lifecycle owner as start.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/start.sh" "$@" --profile nvfp4-nvidia
