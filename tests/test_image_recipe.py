#!/usr/bin/env python3
"""Defaults and overlay recipe-stamp rebuild contract."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
DOCKERFILE = ROOT / "Dockerfile"
ENV_EXAMPLE = ROOT / ".env.example"
C4_CAPACITY_WARMUP = ROOT / "scripts" / "c4-capacity-warmup.sh"


def test_documented_defaults() -> None:
    start = START.read_text()
    example = ENV_EXAMPLE.read_text()
    assert 'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"' in start
    assert 'EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-1}"' in start
    assert 'IMAGE="${IMAGE:-sparkglm:local}"' in start
    assert 'SPEC_METHOD="${SPEC_METHOD:-mtp}"' in start
    assert 'LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"' in start
    assert 'EXL3_GROUPED_PREFILL_K4="${EXL3_GROUPED_PREFILL_K4:-0}"' in start
    assert "MAX_NUM_BATCHED_TOKENS=7168" in example
    assert re.search(r"^EXL3_FAT_KERNEL=1$", example, re.M)
    assert re.search(r"^IMAGE=sparkglm:local$", example, re.M)
    assert re.search(r"^SPEC_METHOD=mtp$", example, re.M)
    assert re.search(r"^LANGUAGE_MODEL_ONLY=1$", example, re.M)
    assert re.search(r"^EXL3_GROUPED_PREFILL_K4=0$", example, re.M)


def test_docker_copy_sources_exist_and_ignored_files_are_not_required() -> None:
    dockerfile = DOCKERFILE.read_text()
    copies: list[str] = []
    for logical in dockerfile.replace("\\\n", " ").splitlines():
        stripped = logical.strip()
        if not stripped.startswith("COPY ") or stripped.startswith("COPY --from="):
            continue
        words = stripped.split()[1:]
        copies.extend(words[:-1])
    missing = [path for path in copies if not (ROOT / path).exists()]
    assert not missing, f"Dockerfile COPY sources missing from build context: {missing}"
    assert "refusal_direction_glm53_bf_oproj.pt" not in dockerfile
    assert "refusal_direction_glm53_dealign_late.pt" not in dockerfile
    for name in ("exl3_decode_moe.cu", "exl3_decode_moe.cuh"):
        assert f"COPY overlay/{name} " in dockerfile


def test_recipe_stamp_wiring() -> None:
    start = START.read_text()
    dockerfile = DOCKERFILE.read_text()
    assert "overlay_recipe_hash() {" in start
    assert '"$SCRIPT_DIR/patches"' in start
    assert "image_recipe_stamp() {" in start
    assert 'SKIP_BUILD:-0' in start
    assert "--build-arg" in start and "GLM53_RECIPE_STAMP" in start
    assert "SPARKGLM_SOURCE_REVISION" in start
    assert "ARG GLM53_RECIPE_STAMP=unknown" in dockerfile
    assert "LABEL glm53.recipe.stamp=${GLM53_RECIPE_STAMP}" in dockerfile
    assert "LABEL glm53.recipe.stamp=${GLM53_RECIPE_STAMP}" in dockerfile
    assert dockerfile.rstrip().endswith(
        "org.opencontainers.image.revision=${SPARKGLM_SOURCE_REVISION}"
    )


def test_overlay_recipe_hash_runs() -> None:
    source = START.read_text()
    begin = source.index("overlay_recipe_hash() {")
    end = source.index("\nimage_recipe_stamp()")
    script = f"SCRIPT_DIR={str(ROOT)!r}\n" + source[begin:end] + "overlay_recipe_hash\n"
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", digest), digest


def test_long_c4_capacity_warmup_is_fatal_and_checks_residency() -> None:
    start = START.read_text()
    warmup = C4_CAPACITY_WARMUP.read_text()
    assert 'GLM53_BOOT_LONG_C4="${GLM53_BOOT_LONG_C4:-1}"' in start
    assert "post_ready_c4_capacity_warmup" in start
    assert '|| die "long C4 capacity warmup failed"' in start
    assert 'if [ "$max_running" -lt 4 ]' in warmup
    subprocess.run(["bash", "-n", C4_CAPACITY_WARMUP], check=True)


if __name__ == "__main__":
    test_documented_defaults()
    test_docker_copy_sources_exist_and_ignored_files_are_not_required()
    test_recipe_stamp_wiring()
    test_overlay_recipe_hash_runs()
    test_long_c4_capacity_warmup_is_fatal_and_checks_residency()
    print("image recipe tests: PASS")
