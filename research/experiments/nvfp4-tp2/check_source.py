#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check copied installer compatibility/idempotence on pinned source, without GPUs."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parent
BASE = "487ecf187d3dfe74d2cf6119a92881dba403c219"


def check(source: Path) -> None:
    with tempfile.TemporaryDirectory() as temp:
        site, opt = Path(temp) / "vllm", Path(temp) / "opt"
        opt.mkdir()
        for relative in ("model_executor/models/qwen3_dflash.py", "model_executor/models/registry.py",
                         "v1/worker/gpu/spec_decode/dflash/utils.py", "v1/worker/gpu/spec_decode/__init__.py"):
            path = site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(["git", "-C", str(source), "show", BASE + ":vllm/" + relative]))
        for name in ("qwen3_dflash2.py", "dflash2_speculator.py"):
            shutil.copyfile(ROOT / "overlay" / name, opt / name)

        def apply(path):
            spec = importlib.util.spec_from_file_location("patch_probe", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.SITE, mod.OPT = site, opt
            mod.main()

        apply(ROOT / "overlay/patch_dflash2.py")
        shutil.copyfile(EXP / "overlay/qwen3_dflash2.py", opt / "qwen3_dflash2.py")
        apply(EXP / "overlay/patch_dflash2.py")
        before = {str(p.relative_to(site)): p.read_bytes() for p in site.rglob("*.py")}
        apply(EXP / "overlay/patch_dflash2.py")
        if before != {str(p.relative_to(site)): p.read_bytes() for p in site.rglob("*.py")}:
            raise ValueError("second patch application changed source")
        print("PASS: MXFP8 installer applies to pinned vLLM + SparkGLM DFlash2 and is idempotent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    check(parser.parse_args().vllm_source)
