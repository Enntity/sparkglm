#!/usr/bin/env python3
"""Regression test for caller overrides that must win over ``.env``."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_max_num_seqs_inline_override_wins() -> None:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            preamble
            + '\nprintf "MAX_NUM_SEQS=%s\\n" "${MAX_NUM_SEQS:-unset}"\n'
        )
        script.chmod(0o755)
        (tmp / ".env").write_text("MAX_NUM_SEQS=2\n")

        env = os.environ.copy()
        env["MAX_NUM_SEQS"] = "4"
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == "MAX_NUM_SEQS=4"


def _run_preamble(env_file: str, caller: dict[str, str], probe: str) -> str:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(preamble + probe)
        script.chmod(0o755)
        (tmp / ".env").write_text(env_file)

        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "GLM53_INDEXER_WORKSPACE",
                "GLM53_MIXED_PREFILL_CHUNK",
                "GLM53_SPINWAIT_MS",
            )
        }
        env.update(caller)
        result = subprocess.run(
            ["bash", str(script)], check=True, capture_output=True, text=True, env=env
        )
    return result.stdout.strip()


def test_indexer_workspace_caller_capture_is_setness_aware() -> None:
    """An explicitly EMPTY caller value must not be swallowed by ``.env``.

    ``GLM53_INDEXER_WORKSPACE=`` is an operator error; the enum guard has to see
    it. A ``[ -n "$_cli_..." ]`` restore would silently hand back the ``.env``
    value instead, so the capture uses the ``${VAR+1}`` setness probe.
    """
    probe = '\nprintf "V=[%s]\\n" "${GLM53_INDEXER_WORKSPACE-UNSET}"\n'
    env_file = "GLM53_INDEXER_WORKSPACE=rightsize\n"

    # Caller silent: .env wins.
    assert _run_preamble(env_file, {}, probe) == "V=[rightsize]"
    # Caller sets a real value: caller wins (the pre-existing contract).
    assert _run_preamble(
        env_file, {"GLM53_INDEXER_WORKSPACE": "stock"}, probe
    ) == "V=[stock]"
    # Caller sets it EMPTY: the empty value survives to the guard.
    assert _run_preamble(
        env_file, {"GLM53_INDEXER_WORKSPACE": ""}, probe
    ) == "V=[]"
    # ... and with no .env value either.
    assert _run_preamble("", {"GLM53_INDEXER_WORKSPACE": ""}, probe) == "V=[]"
    # Unset on both sides stays unset until the configuration default.
    assert _run_preamble("", {}, probe) == "V=[UNSET]"


def test_spinwait_caller_capture_is_setness_aware() -> None:
    probe = '\nprintf "V=[%s]\\n" "${GLM53_SPINWAIT_MS-UNSET}"\n'
    env_file = "GLM53_SPINWAIT_MS=16\n"

    assert _run_preamble(env_file, {}, probe) == "V=[16]"
    assert _run_preamble(
        env_file, {"GLM53_SPINWAIT_MS": "stock"}, probe
    ) == "V=[stock]"
    assert _run_preamble(
        env_file, {"GLM53_SPINWAIT_MS": ""}, probe
    ) == "V=[]"
    assert _run_preamble("", {"GLM53_SPINWAIT_MS": ""}, probe) == "V=[]"
    assert _run_preamble("", {}, probe) == "V=[UNSET]"


def test_mixed_prefill_caller_override_wins() -> None:
    probe = '\nprintf "V=[%s]\\n" "${GLM53_MIXED_PREFILL_CHUNK-UNSET}"\n'
    env_file = "GLM53_MIXED_PREFILL_CHUNK=0\n"

    assert _run_preamble(env_file, {}, probe) == "V=[0]"
    assert _run_preamble(
        env_file, {"GLM53_MIXED_PREFILL_CHUNK": "skip"}, probe
    ) == "V=[skip]"


def test_fat_kernel_subfeature_overrides_win() -> None:
    probe = (
        '\nprintf "TILE=%s PAIR=%s ACT=%s GROUPED=%s\\n" '
        '"${EXL3_FAT_TILE_M-UNSET}" "${EXL3_FAT_PAIR-UNSET}" '
        '"${EXL3_FAT_FUSED_ACT-UNSET}" "${EXL3_GROUPED_PREFILL_K4-UNSET}"\n'
    )
    env_file = (
        "EXL3_FAT_TILE_M=128\nEXL3_FAT_PAIR=0\nEXL3_FAT_FUSED_ACT=1\n"
        "EXL3_GROUPED_PREFILL_K4=0\n"
    )
    assert _run_preamble(
        env_file,
        {
            "EXL3_FAT_TILE_M": "64",
            "EXL3_FAT_PAIR": "1",
            "EXL3_FAT_FUSED_ACT": "0",
            "EXL3_GROUPED_PREFILL_K4": "1",
        },
        probe,
    ) == "TILE=64 PAIR=1 ACT=0 GROUPED=1"


def test_decode_coop_overrides_win_and_defaults_are_materialized() -> None:
    probe = (
        '\nprintf "COOP=%s MAX=%s\\n" '
        '"${EXL3_DECODE_COOP_K4-UNSET}" '
        '"${EXL3_DECODE_COOP_MAX_TOKENS-UNSET}"\n'
    )
    env_file = "EXL3_DECODE_COOP_K4=0\nEXL3_DECODE_COOP_MAX_TOKENS=8\n"
    assert _run_preamble(
        env_file,
        {
            "EXL3_DECODE_COOP_K4": "1",
            "EXL3_DECODE_COOP_MAX_TOKENS": "16",
        },
        probe,
    ) == "COOP=1 MAX=16"
    source = (ROOT / "start.sh").read_text()
    assert 'EXL3_DECODE_COOP_K4="${EXL3_DECODE_COOP_K4:-1}"' in source
    assert (
        'EXL3_DECODE_COOP_MAX_TOKENS="${EXL3_DECODE_COOP_MAX_TOKENS:-16}"'
        in source
    )


def test_tiny_model_identity_overrides_win() -> None:
    probe = (
        '\nprintf "%s|%s|%s|%s|%s|%s|%s|%s\\n" "$MODEL" "$MODEL_FALLBACK" '
        '"$MODEL_CACHE_NAME" "$MODEL_REVISION" "$MODEL_FALLBACK_REVISION" '
        '"$DFLASH_REVISION" "$SERVED_MODEL_NAME" "$GLM53_BOOT_SHAPE_WARMUP"\n'
    )
    env_file = (
        "MODEL=production/model\n"
        "MODEL_FALLBACK=production/fallback\n"
        "MODEL_CACHE_NAME=models--production\n"
        "MODEL_REVISION=production-model-revision\n"
        "MODEL_FALLBACK_REVISION=production-fallback-revision\n"
        "DFLASH_REVISION=production-dflash-revision\n"
        "SERVED_MODEL_NAME=production\n"
        "GLM53_BOOT_SHAPE_WARMUP=1\n"
    )
    result = _run_preamble(
        env_file,
        {
            "MODEL": "sparkglm/tinyglm",
            "MODEL_FALLBACK": "sparkglm/tinyglm",
            "MODEL_CACHE_NAME": "models--sparkglm--tinyglm",
            "MODEL_REVISION": "tinyglm-v1",
            "MODEL_FALLBACK_REVISION": "tinyglm-v1",
            "DFLASH_REVISION": "test-dflash-revision",
            "SERVED_MODEL_NAME": "tinyGLM-5.3-EXL3",
            "GLM53_BOOT_SHAPE_WARMUP": "0",
        },
        probe,
    )
    assert result == (
        "sparkglm/tinyglm|sparkglm/tinyglm|models--sparkglm--tinyglm|tinyglm-v1|"
        "tinyglm-v1|test-dflash-revision|tinyGLM-5.3-EXL3|0"
    )


if __name__ == "__main__":
    test_max_num_seqs_inline_override_wins()
    test_indexer_workspace_caller_capture_is_setness_aware()
    test_spinwait_caller_capture_is_setness_aware()
    test_fat_kernel_subfeature_overrides_win()
    test_decode_coop_overrides_win_and_defaults_are_materialized()
    test_tiny_model_identity_overrides_win()
    print("start.sh caller override regression OK")
