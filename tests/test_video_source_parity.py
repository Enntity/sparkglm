#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source identity must be checked, not inferred from optimization flags."""
import hashlib
import argparse
import importlib.util
import json
import subprocess
import sys
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


def test_candidate_is_explicit_and_fail_closed():
    before = "a" * 64
    after = "b" * 64
    name = "vllm/test.py"
    manifest = {"python_files": {name: before}, "native_source": {},
                "exl3_compile_inputs": {}, "allowed_nonfunctional_differences": {},
                "allowed_exl3_license_differences": {}}
    change = {"reference_sha256": before, "candidate_sha256": after, "reason": "test-only experiment"}
    declaration = {"schema": "sparkglm.candidate-sources/v1", "changes": {"python": {name: change}}}
    expected, allowed = parity.candidate_expectations(manifest, declaration, "python", {name: before}, {})
    assert expected == {name: after} and allowed == {}
    assert manifest["python_files"][name] == before  # never rewrite the baseline
    for field, bad in (("reference_sha256", "c" * 64), ("candidate_sha256", "bad"), ("reason", "")):
        invalid = json.loads(json.dumps(declaration))
        invalid["changes"]["python"][name][field] = bad
        try:
            parity.candidate_expectations(manifest, invalid, "native", {}, {})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid {field}")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "vllm").mkdir()
        path = root / name
        path.write_text("candidate\n")
        change["candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        expected, allowed = parity.candidate_expectations(manifest, declaration, "python", {name: before}, {})
        assert parity.verify(root, expected, allowed, inventory=True)["passed"]
        assert not parity.verify(root, {name: before}, inventory=True)["passed"]
        path.write_text("undeclared change\n")
        assert not parity.verify(root, expected, allowed)["passed"]
        path.unlink()
        assert not parity.verify(root, expected, allowed)["passed"]
    frozen = (ROOT / "provenance/video-source-parity.json").read_bytes()
    command = [sys.executable, str(ROOT / "scripts/build_candidate.py"),
               "--manifest", str(ROOT / "provenance/candidate-sources.example.json"),
               "--tag", "sparkglm-candidate:test", "--print-dockerfile"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert result.stdout.count("--candidate-manifest /opt/sparkglm-candidate-sources.json") == 3
    assert result.stdout.count("COPY provenance/candidate-sources.example.json /opt/sparkglm-candidate-sources.json") == 2
    assert 'org.enntity.sparkglm.build-profile="candidate"' in result.stdout
    assert frozen == (ROOT / "provenance/video-source-parity.json").read_bytes()
    assert '--candidate-manifest' not in (ROOT / "Dockerfile").read_text()
    for kind in ("native", "python", "exl3"):
        assert f"--kind {kind}" in result.stdout
    command[command.index("sparkglm-candidate:test")] = "sparkglm:local"
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_contributor_dependencies_and_context_boundaries():
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    workflow = (ROOT / ".github/workflows/static.yml").read_text()
    assert "pip install -r requirements-dev.txt" in contributing
    assert "pip install -r requirements-dev.txt" in workflow
    assert "Jinja2>=3.1,<4" in (ROOT / "requirements-dev.txt").read_text()
    excluded = (ROOT / ".dockerignore").read_text().splitlines()
    for pattern in (".git", ".env", ".venv", "**/*.safetensors", "**/*.pt", "**/*.pem"):
        assert pattern in excluded


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
    test_candidate_is_explicit_and_fail_closed()
    test_contributor_dependencies_and_context_boundaries()
    test_replay_cache_namespace_does_not_change_video_request_inputs()
    print("video source parity gate: PASS")
