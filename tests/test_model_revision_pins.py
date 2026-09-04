#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for immutable model and drafter cache selection."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "start.sh").read_text()


def function(name: str) -> str:
    marker = f"{name}() {{\n"
    start = SOURCE.index(marker)
    lines = SOURCE[start:].splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        result.append(line)
        if line == "}\n" or line == "}":
            return "".join(result)
    raise AssertionError(f"unterminated shell function: {name}")


FUNCTIONS = "\n".join(
    function(name)
    for name in (
        "count_shards",
        "count_revision_shards",
        "ensure_refs_main",
        "ensure_dflash_refs_main",
    )
)


def run(body: str, directory: Path) -> subprocess.CompletedProcess[str]:
    script = f"""set -euo pipefail
log() {{ :; }}
die() {{ echo "$*" >&2; exit 7; }}
{FUNCTIONS}
TEST_ROOT={directory}
{body}
"""
    return subprocess.run(["bash"], input=script, text=True, capture_output=True)


def test_exact_revisions_are_selected() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        result = run(
            """
MODEL_PATH="$TEST_ROOT/model"
DFLASH_PATH="$TEST_ROOT/dflash"
ACTIVE_MODEL_REVISION=model-pin
DFLASH_REVISION=dflash-pin
mkdir -p "$MODEL_PATH/snapshots/model-pin" "$MODEL_PATH/snapshots/other"
mkdir -p "$DFLASH_PATH/snapshots/dflash-pin" "$DFLASH_PATH/snapshots/other"
touch "$MODEL_PATH/snapshots/model-pin/a.safetensors"
touch "$MODEL_PATH/snapshots/other/a.safetensors" "$MODEL_PATH/snapshots/other/b.safetensors"
ensure_refs_main
ensure_dflash_refs_main
test "$(cat "$MODEL_PATH/refs/main")" = model-pin
test "$(cat "$DFLASH_PATH/refs/main")" = dflash-pin
test "$(count_revision_shards "$MODEL_PATH" model-pin)" = 1
""",
            root,
        )
        assert result.returncode == 0, result.stderr


def test_missing_pin_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        result = run(
            """
MODEL_PATH="$TEST_ROOT/model"
ACTIVE_MODEL_REVISION=required-pin
mkdir -p "$MODEL_PATH/snapshots/other"
ensure_refs_main
""",
            root,
        )
        assert result.returncode == 7
        assert "pinned model revision missing: required-pin" in result.stderr


def test_missing_dflash_pin_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        result = run(
            """
DFLASH_PATH="$TEST_ROOT/dflash"
DFLASH_REVISION=required-dflash-pin
mkdir -p "$DFLASH_PATH/snapshots/other"
ensure_dflash_refs_main
""",
            root,
        )
        assert result.returncode == 7
        assert "pinned DFlash2 revision missing: required-dflash-pin" in result.stderr


if __name__ == "__main__":
    test_exact_revisions_are_selected()
    test_missing_pin_fails_closed()
    test_missing_dflash_pin_fails_closed()
    print("immutable model revision pins OK (3 tests)")
