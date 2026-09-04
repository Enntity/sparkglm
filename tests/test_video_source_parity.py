#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source identity must be checked, not inferred from optimization flags."""
import hashlib
import argparse
import importlib.util
import json
import tempfile
import threading
import time
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
    native_build = (ROOT / "scripts/build-video-native.sh").read_text()
    assert "-U '*_GEN_SCRIPT_HASH_AND_ARCH'" in native_build
    assert docker.index("video-runtime.patch") < docker.index("--kind python")
    runtime = (ROOT / "patches/video/runtime.patch").read_text()
    for name in ("config/attention.py", "layers/sparse_attn_indexer.py",
                 "layers/sparse_attn_indexer_kpool.py", "utils/deep_gemm.py",
                 "core/kv_cache_coordinator.py", "core/sched/scheduler.py",
                 "vllm_flash_attn/__init__.py"):
        assert name in runtime


def test_replay_cache_namespace_does_not_change_video_request_inputs():
    spec = importlib.util.spec_from_file_location("video", ROOT / "benchmarks/four_stream_video.py")
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)
    captured = []
    original = video.urllib.request.urlopen

    def intercept(request, **_kwargs):
        captured.append(json.loads(request.data))
        raise RuntimeError("offline request capture")

    args = argparse.Namespace(
        prompt_style="field-guide", prompt_tokens=16000,
        prompt_salt="current-best-video-20260903-r1", model="glm-5.3-flash-exl3",
        max_tokens=400, logprobs=False, api_key="", base_url="http://unused.invalid",
        stagger_ms=0, timeout_s=1, cache_salt=None,
    )
    try:
        video.urllib.request.urlopen = intercept
        for salt in (None, "isolated-replay"):
            args.cache_salt = salt
            for i in range(4):
                video._stream_one(i, video.TOPICS[i], threading.Barrier(1),
                                  time.monotonic(), args, [None] * 4)
    finally:
        video.urllib.request.urlopen = original
    for original_body, isolated in zip(captured[:4], captured[4:]):
        assert "cache_salt" not in original_body
        assert isolated.pop("cache_salt") == "isolated-replay"
        assert isolated == original_body


if __name__ == "__main__":
    test_gate()
    test_wiring()
    test_replay_cache_namespace_does_not_change_video_request_inputs()
    print("video source parity gate: PASS")
