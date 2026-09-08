#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Standalone NVFP4 by default. LLooM integration is explicitly optional.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--lloom" ]]; then
    shift
    exec python3 "$ROOT/scripts/appliance.py" "$@"
fi
exec python3 "$ROOT/scripts/standalone.py" "$@"
