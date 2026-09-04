#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source tests for the runtime commit-provenance policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_commit_provenance", ROOT / "scripts" / "check_commit_provenance.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_docs_only_commit_is_out_of_scope() -> None:
    assert MODULE.validate_message("plain docs change", ["docs/README.md"]) == []


def test_original_runtime_commit_requires_all_sections() -> None:
    message = """perf: original change

Provenance: original implementation.

Original work:
- GB10-specific scheduling.

Verification:
- tinyGLM passed.
"""
    assert MODULE.validate_message(message, ["overlay/kernel.cu"]) == []
    assert "missing Verification:" in MODULE.validate_message(
        message.replace("Verification:", "Tests:"), ["overlay/kernel.cu"]
    )


def test_external_runtime_commit_requires_exact_source() -> None:
    good = """perf: adapted change

Provenance:
- https://github.com/example/project at 0123456789abcdef0123456789abcdef01234567: adapted.

Original work:
- GB10 integration.

Verification:
- gate passed.
"""
    assert MODULE.validate_message(good, ["start.sh"]) == []
    errors = MODULE.validate_message(good.replace("0123456789abcdef0123456789abcdef01234567", "0123456"), ["start.sh"])
    assert "external provenance needs a full 40-character revision" in errors


if __name__ == "__main__":
    test_docs_only_commit_is_out_of_scope()
    test_original_runtime_commit_requires_all_sections()
    test_external_runtime_commit_requires_exact_source()
    print("commit provenance policy OK (3 tests)")
