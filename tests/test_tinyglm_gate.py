#!/usr/bin/env python3
"""CPU-only comparison tests for the tinyGLM promotion gate."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "benchmarks" / "tinyglm_gate.py"
    spec = importlib.util.spec_from_file_location("tinyglm_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report() -> dict:
    cases = []
    for name in ("decode_c1", "mixed_c4", "long_c2"):
        cases.append(
            {
                "name": name,
                "concurrency": 1,
                "output_tokens": 32,
                "runs": [
                    {
                        "streams": [
                            {
                                "stream": 0,
                                "completion_tokens": 32,
                                "token_ids_count": 32,
                            }
                        ]
                    },
                    {
                        "streams": [
                            {
                                "stream": 0,
                                "completion_tokens": 32,
                                "token_ids_count": 32,
                            }
                        ]
                    },
                ],
                "summary": {
                    "aggregate_tok_s_median": 100.0,
                    "wall_s_median": 1.0,
                    "ttft_s_median": 0.2,
                    "signatures": [[f"{name}-s0"], [f"{name}-s0"]],
                    "deterministic": True,
                },
            }
        )
    return {"schema": 1, "cases": cases}


def test_equal_report_passes() -> None:
    module = _module()
    report = _report()
    assert module.compare_reports(report, deepcopy(report), 5.0) == []


def test_correctness_and_performance_regressions_fail() -> None:
    module = _module()
    baseline = _report()
    candidate = deepcopy(baseline)
    candidate["cases"][0]["summary"]["signatures"][0] = ["changed"]
    candidate["cases"][1]["summary"]["aggregate_tok_s_median"] = 90.0
    candidate["cases"][2]["summary"]["ttft_s_median"] = 0.5
    failures = module.compare_reports(baseline, candidate, 5.0)
    assert any("token IDs changed" in failure for failure in failures)
    assert any("throughput" in failure for failure in failures)
    assert any("TTFT" in failure for failure in failures)


def test_incomplete_or_nondeterministic_report_fails() -> None:
    module = _module()
    report = _report()
    report["cases"][0]["summary"]["deterministic"] = False
    report["cases"][1]["runs"][0]["streams"][0]["token_ids_count"] = 0
    failures = module.validate_report(report)
    assert any("nondeterministic" in failure for failure in failures)
    assert any("token IDs missing" in failure for failure in failures)


if __name__ == "__main__":
    test_equal_report_passes()
    test_correctness_and_performance_regressions_fail()
    test_incomplete_or_nondeterministic_report_fails()
    print("tinyGLM comparison gate tests OK")
