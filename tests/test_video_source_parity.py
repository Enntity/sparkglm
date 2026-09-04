#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source identity must be checked, not inferred from optimization flags."""
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("parity", ROOT / "scripts/check_video_source_parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_gate():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "vllm").mkdir()
        target = root / "vllm/test.py"
        target.write_text("x = 1\n")
        expected = {"vllm/test.py": hashlib.sha256(target.read_bytes()).hexdigest()}
        assert parity.verify(root, expected, inventory=True)["passed"]
        target.write_text("x = 2\n")
        assert not parity.verify(root, expected)["passed"]
        allowed = {"vllm/test.py": {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "reason": "test fixture"}}
        assert parity.verify(root, expected, allowed)["passed"]
        target.unlink()
        assert not parity.verify(root, expected, allowed)["passed"]
        target.write_text("x = 1\n")
        (root / "vllm/extra.py").write_text("pass\n")
        assert not parity.verify(root, expected, inventory=True)["passed"]


def test_wiring():
    manifest = json.loads((ROOT / "provenance/video-source-parity.json").read_text())
    assert len(manifest["python_files"]) == 2423
    assert len(manifest["native_source"]) == 8
    assert len(manifest["exl3_compile_inputs"]) == 6
    assert set(manifest["allowed_nonfunctional_differences"]) == {
        "vllm/_version.py", "vllm/model_executor/layers/quantization/exl3.py",
        "vllm/models/common/ops/fused_qk_rmsnorm.py"}
    docker = (ROOT / "Dockerfile").read_text()
    assert "--kind native" in docker and "--kind python" in docker
    assert "COPY --from=video-native /opt/video-native/vllm/" in docker
    assert docker.index("video-runtime.patch") < docker.index("--kind python")
    runtime = (ROOT / "patches/video/runtime.patch").read_text()
    for name in ("config/attention.py", "layers/sparse_attn_indexer.py",
                 "layers/sparse_attn_indexer_kpool.py", "utils/deep_gemm.py",
                 "core/kv_cache_coordinator.py", "core/sched/scheduler.py",
                 "vllm_flash_attn/__init__.py"):
        assert name in runtime


if __name__ == "__main__":
    test_gate()
    test_wiring()
    print("video source parity gate: PASS")
