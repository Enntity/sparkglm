#!/usr/bin/env python3
"""Structural tests for the weightless tinyGLM fixture."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_tinyglm.py"


def _module():
    spec = importlib.util.spec_from_file_location("make_tinyglm", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tinyglm_preserves_kernel_selecting_dimensions() -> None:
    module = _module()
    config = module.model_config(experts=16, vocab_size=256, max_length=32768)
    text = config["text_config"]
    assert config["architectures"] == ["Glm5NextForCausalLM"]
    assert text["hidden_size"] == 4096
    assert text["moe_intermediate_size"] == 2048
    assert text["num_experts_per_tok"] == 8
    assert text["layer_types"] == [
        "linear_attention",
        "deepseek_sparse_attention",
    ]
    assert text["mlp_layer_types"] == ["sparse", "sparse"]
    assert text["q_lora_rank"] == 1536
    assert text["kv_lora_rank"] == 512
    assert text["index_topk"] == 2048
    assert text["index_kpool"] == 4
    assert text["mhc"] is True
    assert text["swiglu_limit"] == 10.0
    assert config["quantization_config"]["bits"] == 4


def test_tinyglm_builds_self_contained_weightless_snapshot() -> None:
    module = _module()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "models--sparkglm--tinyglm"
        snapshot = module.build(root, experts=16, vocab_size=256, max_length=32768)
        assert (root / "refs" / "main").read_text().strip() == snapshot.name
        assert snapshot.name == "tinyglm-v1-e16-v256-l32768"
        expected = {
            "config.json",
            "quantization_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "README.md",
        }
        assert {path.name for path in snapshot.iterdir()} == expected
        assert not list(snapshot.glob("*.safetensors"))
        tokenizer = json.loads((snapshot / "tokenizer.json").read_text())
        assert len(tokenizer["model"]["vocab"]) == 256


def test_explicit_candidate_image_survives_reference_env() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "scripts").mkdir()
        script = root / "scripts/tinyglm.sh"
        script.write_text((ROOT / "scripts/tinyglm.sh").read_text())
        (root / ".env").write_text("IMAGE=sparkglm:local\n")
        (root / "scripts/make_tinyglm.py").write_text(
            'print("/fixture/snapshots/tinyglm-v1-e16-v256-l32768")\n')
        start = root / "start.sh"
        start.write_text('#!/usr/bin/env bash\nprintf "%s %s %s\\n" "$IMAGE" "$MODEL_REVISION" "$SPARKGLM_TINY_DUMMY"\n')
        start.chmod(0o755)
        env = dict(os.environ)
        env["IMAGE"] = "sparkglm-candidate:regression-test"
        result = subprocess.run(["bash", str(script), "restart"], env=env,
                                check=True, capture_output=True, text=True)
        assert result.stdout.strip() == "sparkglm-candidate:regression-test tinyglm-v1-e16-v256-l32768 1"
        env.pop("IMAGE")
        result = subprocess.run(["bash", str(script), "restart"], env=env,
                                check=True, capture_output=True, text=True)
        assert result.stdout.startswith("sparkglm:local ")


if __name__ == "__main__":
    launcher = (ROOT / "scripts/tinyglm.sh").read_text()
    assert 'snapshot="$(build)"' in launcher
    assert 'export MODEL_REVISION="${snapshot##*/}"' in launcher
    test_tinyglm_preserves_kernel_selecting_dimensions()
    test_tinyglm_builds_self_contained_weightless_snapshot()
    test_explicit_candidate_image_survives_reference_env()
    print("tinyGLM structural tests OK")
