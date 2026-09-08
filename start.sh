#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Default: measured NVFP4 profile under LLooM. EXL3 remains explicitly selectable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT/scripts/appliance.py" "$@"
