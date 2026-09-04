#!/usr/bin/env bash
# Promotion gate for GLM kernel, scheduler, graph, and TP2 experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

static_gate() {
    scripts/check.sh quick
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
