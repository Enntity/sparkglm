#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run only after a maintainer reserves an idle GPU; never stop serving containers.
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo "usage: run-g1.sh REFERENCE_IMAGE CANDIDATE_IMAGE NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
reference="$1"
candidate="$2"
output="$3"
if [ "${SPARKGLM_GPU_RESERVED:-}" != "1" ]; then
    echo "Reserve an idle GPU first, then explicitly set SPARKGLM_GPU_RESERVED=1." >&2
    exit 2
fi
case "$candidate" in
    sparkglm-candidate:*) ;;
    *) echo "Candidate must use a separate sparkglm-candidate tag." >&2; exit 2 ;;
esac
reference_id="$(docker image inspect --format '{{.Id}}' "$reference")"
candidate_id="$(docker image inspect --format '{{.Id}}' "$candidate")"
if [ "$reference_id" = "$candidate_id" ]; then
    echo "Reference and candidate resolve to the same image." >&2
    exit 2
fi
# No overwriting earlier receipts. IDs rather than tags pin every subsequent run.
mkdir "$output"
output="$(cd "$output" && pwd)"
source_hash() {
    docker run --rm --network none --entrypoint sha256sum "$1" \
        /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu | awk '{print $1}'
}
reference_hash="$(source_hash "$reference_id")"
candidate_hash="$(source_hash "$candidate_id")"
python3 - "$root" "$reference_hash" "$candidate_hash" "${SPARKGLM_EPILOGUE_VARIANT:-direct}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
assert sys.argv[4] in ('direct', 'coalesced'), 'unknown epilogue variant'
name = 'sources.json' if sys.argv[4] == 'direct' else 'sources-coalesced.json'
manifest = json.loads((root / 'research/experiments/exl3-direct-epilogue' / name).read_text())
change = manifest['changes']['exl3']['exl3-fat-kernel/exl3_fat_gemm.cu']
assert sys.argv[2] == change['reference_sha256'], 'wrong reference source'
assert sys.argv[3] == change['candidate_sha256'], 'wrong candidate source'
PY
for pair in 1 2 3; do
    if [ "$pair" -eq 2 ]; then arms="candidate reference"; else arms="reference candidate"; fi
    for arm in $arms; do
        if [ "$arm" = reference ]; then image_id="$reference_id"; source_sha="$reference_hash";
        else image_id="$candidate_id"; source_sha="$candidate_hash"; fi
        docker run --rm --network none --gpus all --shm-size 1g \
            -v "$root:/workspace:ro" -w /workspace --entrypoint python3 "$image_id" \
            research/experiments/exl3-direct-epilogue/bench.py \
            --arm "$arm" --image-id "$image_id" --source-sha256 "$source_sha" \
            --seed "$((20260905 + pair))" \
            > "$output/$arm-pair-$pair.json" 2> "$output/$arm-pair-$pair.stderr"
    done
done
echo "G1 receipts saved. Review every pair; these are not endpoint or quality results."
