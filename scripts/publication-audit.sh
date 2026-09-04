#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

failures=0

fail() {
    echo "FAIL: $*" >&2
    failures=$((failures + 1))
}

require_file() {
    [ -f "$1" ] || fail "required file missing: $1"
}

require_file LICENSE
require_file NOTICE
require_file LICENSES/Apache-2.0.txt
require_file LICENSES/MIT-Mia.txt
require_file LICENSES/MIT-ExLlamaV3.txt
require_file LICENSES/AGPL-3.0-only.txt
require_file LICENSES/MIT-FlashKDA.txt
require_file LICENSES/NOTICE-Reederey.txt
require_file LICENSES/GLM-5.3.txt
require_file docs/ATTRIBUTION.md
require_file docs/PUBLICATION_REVIEW.md
require_file docs/METHODOLOGY.md
require_file docs/QUALIFICATION.md
require_file results/README.md
require_file results/index.json
require_file schemas/qualification-v1.schema.json
require_file research/atlas/atlas-glm53.patch

prohibited_files="$({
    git ls-files | grep -E '(^|/)(\.env|\.DS_Store)$|\.(pt|safetensors|gguf|bin|so|pyc|key|pem|p12|mp4|mov|mkv)$' || true
} | sed '/^$/d')"
if [ -n "$prohibited_files" ]; then
    echo "$prohibited_files" >&2
    fail "prohibited generated, model, credential, or media files are tracked"
fi

if git grep -I -l -E \
    'BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}' \
    -- . >/tmp/sparkglm-publication-secret-files.txt; then
    cat /tmp/sparkglm-publication-secret-files.txt >&2
    fail "credential-like material found"
fi

if git grep -I -l -E \
    '/Users/jmac|/home/enntitysparkadmin|ennspark0[12]' \
    -- . ':!scripts/publication-audit.sh' \
    >/tmp/sparkglm-publication-local-files.txt; then
    cat /tmp/sparkglm-publication-local-files.txt >&2
    fail "private workstation path or Spark hostname found"
fi

for pin in \
    eb0469fbb2b49fd7c025f594a3339a121e58f7a9 \
    487ecf187d3dfe74d2cf6119a92881dba403c219 \
    c5d9c657966ffeeaa9353f0cc899f18629da4a13 \
    0c03250cd7176a2fef9cbbf9329fed08c8750e7d \
    bdcccc2ca91eba084aac94a059e3b0f4a5d556dd \
    775cb3655e29a3735f4f58faa540608f9427bf51; do
    grep -R -q "$pin" docs research SPARKGLM.md || fail "provenance pin not documented: $pin"
done

grep -R -q 'aca966e4e02791568aa6a4ced368624b3d897f42' \
    docs NOTICE || fail "pinned Z.ai chat-template revision is undocumented"

while IFS= read -r json_file; do
    python3 -m json.tool "$json_file" >/dev/null || fail "invalid JSON: $json_file"
done < <(git ls-files '*.json')

while IFS= read -r shell_file; do
    bash -n "$shell_file" || fail "invalid shell syntax: $shell_file"
done < <(git ls-files '*.sh')

python3 scripts/qualification.py verify-all \
    || fail "qualification records are invalid or the result index is stale"

if ! grep -q 'libatlas_glm53_flash_kda.so' research/atlas/README.md; then
    fail "Atlas binary exclusion is undocumented"
fi

if grep -q '^GIT binary patch$' research/atlas/atlas-glm53.patch; then
    fail "Atlas archive unexpectedly contains a binary patch"
fi

if [ "$failures" -ne 0 ]; then
    echo "publication audit failed with $failures finding(s)" >&2
    exit 1
fi

echo "publication audit passed"
