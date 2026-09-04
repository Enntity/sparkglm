#!/usr/bin/env bash
# Promotion gate for GLM kernel, scheduler, graph, and TP2 experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

static_gate() {
    python3 tests/test_tinyglm.py
    python3 tests/test_numeric_config.py
    python3 tests/test_start_overrides.py
    python3 tests/test_tinyglm_gate.py
    python3 -m py_compile \
        overlay/exl3.py \
        scripts/make_tinyglm.py \
        benchmarks/exl3_decode_moe_ab.py \
        benchmarks/exl3_grouped_prefill_ab.py \
        benchmarks/tinyglm_smoke.py \
        benchmarks/tinyglm_gate.py \
        benchmarks/tinyglm_prefill_gate.py
    bash -n start.sh scripts/tinyglm.sh scripts/tinyglm-gate.sh
    git diff --check
    echo "tinyGLM static gate: PASS"
}

command="${1:-static}"
case "$command" in
    static)
        static_gate
        ;;
    record)
        output="${2:-.tinyglm-gates/baseline.json}"
        static_gate
        python3 benchmarks/tinyglm_gate.py \
            --endpoint "${TINYGLM_ENDPOINT:-http://127.0.0.1:8888}" \
            --repetitions "${TINYGLM_GATE_REPETITIONS:-3}" \
            --output "$output"
        ;;
    compare)
        baseline="${2:?usage: scripts/tinyglm-gate.sh compare BASELINE [OUTPUT]}"
        output="${3:-.tinyglm-gates/candidate.json}"
        static_gate
        python3 benchmarks/tinyglm_gate.py \
            --endpoint "${TINYGLM_ENDPOINT:-http://127.0.0.1:8888}" \
            --repetitions "${TINYGLM_GATE_REPETITIONS:-3}" \
            --max-regression-pct "${TINYGLM_MAX_REGRESSION_PCT:-5}" \
            --baseline "$baseline" \
            --output "$output"
        ;;
    *)
        echo "usage: scripts/tinyglm-gate.sh [static|record [OUT]|compare BASELINE [OUT]]" >&2
        exit 2
        ;;
esac
