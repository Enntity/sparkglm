#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for full-model workload calibration and fail-closed assertions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def harness():
    path = ROOT / "benchmarks" / "staggered_openai.py"
    spec = importlib.util.spec_from_file_location("staggered_openai", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prompt_calibration_uses_serving_tokenizer() -> None:
    module = harness()
    calls: list[str] = []

    def fake_count(prompt: str, _args: object) -> int:
        calls.append(prompt)
        return 23 + prompt.count(" alpha")

    module.token_count = fake_count
    prompt, count = module.exact_isolation_prompt(
        0, 16384, "pair-c1-16k-r1", SimpleNamespace()
    )
    assert count == 16384
    assert prompt.count(" alpha") == 16384 - 23
    assert module.isolation_marker(0) in prompt
    assert prompt.endswith("FINAL-FACT")
    assert len(calls) < 40


def test_full_gate_enforces_contract() -> None:
    source = (ROOT / "scripts" / "full-model-gate.sh").read_text()
    for required in (
        "--exact-prompt-tokens",
        'request["prompt_tokens"] == expected_prompt_tokens',
        'request["completion_tokens"] == 400',
        'request["own_request_marker"] is True',
        'request["foreign_request_markers"] == []',
        "SPARKGLM_IMAGE_DIGESTS",
        "capture_server_manifest.py",
        "SPARKGLM_FULL_REPETITION_ID",
    ):
        assert required in source


if __name__ == "__main__":
    test_exact_prompt_calibration_uses_serving_tokenizer()
    test_full_gate_enforces_contract()
    print("benchmark contract tests: PASS")
