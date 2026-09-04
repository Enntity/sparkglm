#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

QUICK_TESTS=(
    tests/test_bringup_robustness.py
    tests/test_chat_template.py
    tests/test_image_recipe.py
    tests/test_indexer_workspace.py
    tests/test_kpool_tail_slotmap.py
    tests/test_local_links.py
    tests/test_numeric_config.py
    tests/test_spinwait_patch.py
    tests/test_start_overrides.py
    tests/test_suppress_stops.py
    tests/test_tinyglm.py
    tests/test_tinyglm_gate.py
    tests/test_warm_restart_stdout.py
    tests/test_xgrammar_termination.py
    tests/test_qualification.py
)

CONTAINER_TESTS=(
    tests/test_ablit.py
    tests/test_exl3_overlay.py
    tests/test_hybrid_prefix_hit.py
    tests/test_scheduler_decode_floor.py
)

LIVE_TESTS=(tests/test_mixed_prefill_decode.py)

check_inventory() {
    local difference
    difference="$(comm -3 \
        <(printf '%s\n' "${QUICK_TESTS[@]}" "${CONTAINER_TESTS[@]}" "${LIVE_TESTS[@]}" | sort) \
        <(find tests -maxdepth 1 -name 'test_*.py' -print | sort))"
    if [ -n "$difference" ]; then
        echo "test gate inventory is incomplete:" >&2
        echo "$difference" >&2
        return 1
    fi
}

quick() {
    check_inventory
    for test_file in "${QUICK_TESTS[@]}"; do
        echo "==> $test_file"
        python3 "$test_file"
    done
    python3 -m py_compile \
        overlay/*.py scripts/*.py benchmarks/*.py tests/*.py
    while IFS= read -r shell_file; do
        bash -n "$shell_file"
    done < <(git ls-files '*.sh')
    python3 scripts/qualification.py verify-all
    git diff --check
    echo "SparkGLM quick gate: PASS"
}

container() {
    check_inventory
    for test_file in "${CONTAINER_TESTS[@]}"; do
        echo "==> $test_file"
        python3 "$test_file"
    done
    echo "SparkGLM container gate: PASS"
}

live() {
    check_inventory
    for test_file in "${LIVE_TESTS[@]}"; do
        echo "==> $test_file"
        python3 "$test_file"
    done
    echo "SparkGLM live endpoint gate: PASS"
}

publication() {
    scripts/publication-audit.sh
}

command="${1:-quick}"
case "$command" in
    quick) quick ;;
    container) container ;;
    live) live ;;
    publication) publication ;;
    all)
        quick
        publication
        ;;
    *)
        echo "usage: scripts/check.sh [quick|container|live|publication|all]" >&2
        exit 2
        ;;
esac
