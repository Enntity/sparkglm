#!/usr/bin/env python3
"""Anchor checks for the bring-up robustness patches in start.sh.

Same static-analysis style as test_warm_restart_stdout.py: the launcher is a
generated-heredoc-heavy script, so these tests pin the markers that keep the
robustness behaviours wired (worker death detection, revision-keyed sync
marker, HF CLI fallback, worker cache writability preflight).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "start.sh").read_text()


def test_worker_death_detection_wired() -> None:
    src = _source()
    assert "worker_fail=0" in src, "worker death detection missing from wait_for_health"
    assert '[ "$worker_fail" -ge 3 ]' in src, "3-strike tolerance missing"
    assert "not running on ${WORKER_SSH}" in src, "worker death message missing"
    assert 'dead_side="worker"' in src and 'dead_side="head"' in src


def test_sync_revision_marker_wired() -> None:
    src = _source()
    assert ".glm53-exl3-synced" in src, "revision marker file missing"
    assert "FORCE_SYNC" in src, "FORCE_SYNC escape hatch missing"
    assert 'refs/main' in src, "marker must key on the snapshot commit (refs/main)"
    # both weights and DFlash2 go through the marker-checked helper
    assert src.count("sync_repo_to_worker ") >= 2


def test_hf_cli_fallback_wired() -> None:
    src = _source()
    assert "resolve_hf_bin()" in src, "resolve_hf_bin helper missing"
    assert src.count("resolve_hf_bin || die") == 3, "expected 3 call sites (weights, dflash, download-only)"
    assert "huggingface_hub.commands.huggingface_cli" in src, "python fallback missing"
    assert '"${HF_BIN_CMD[@]}" download' in src, "hf_download_repo must use the resolved array"


def test_worker_cache_writability_preflight_wired() -> None:
    src = _source()
    assert "worker cannot write $WORKER_CACHE_DIR/hub" in src
    assert "test -w '$WORKER_CACHE_DIR/hub'" in src
    assert 'WORKER_CACHE_DIR="${WORKER_CACHE_DIR:-$WORKER_HOME/.cache/huggingface}"' in src


def _run_build_headroom(
    mem_available_kib: int, minimum_gib: int
) -> subprocess.CompletedProcess[str]:
    src = _source()
    begin = src.index("build_mem_available_kib() {")
    end = src.index("\nbuild_image() {", begin)
    functions = src[begin:end]
    with tempfile.TemporaryDirectory() as tmp:
        meminfo = Path(tmp) / "meminfo"
        meminfo.write_text(
            f"MemTotal: 131072000 kB\nMemAvailable: {mem_available_kib} kB\n"
        )
        script = (
            "warn() { printf 'WARN: %s\\n' \"$*\" >&2; }\n"
            "die() { printf 'ERROR: %s\\n' \"$*\" >&2; exit 1; }\n"
            "docker() { :; }\n"
            f"SPARKGLM_MEMINFO_PATH={str(meminfo)!r}\n"
            f"BUILD_MIN_MEM_GIB={minimum_gib}\n"
            + functions
            + "\nassert_build_headroom\n"
        )
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True
        )


def test_build_headroom_guard() -> None:
    enough = _run_build_headroom(40 * 1024 * 1024, 32)
    assert enough.returncode == 0, enough.stderr

    low = _run_build_headroom(8 * 1024 * 1024, 32)
    assert low.returncode != 0
    assert "refusing native image build" in low.stderr

    explicit_override = _run_build_headroom(1, 0)
    assert explicit_override.returncode == 0, explicit_override.stderr

    src = _source()
    build = src[src.index("build_image() {") :]
    assert "assert_build_headroom" in build
    assert 'BUILD_MIN_MEM_GIB="${BUILD_MIN_MEM_GIB:-32}"' in src


if __name__ == "__main__":
    test_worker_death_detection_wired()
    test_sync_revision_marker_wired()
    test_hf_cli_fallback_wired()
    test_worker_cache_writability_preflight_wired()
    test_build_headroom_guard()
    print("start.sh bring-up robustness anchors OK")
