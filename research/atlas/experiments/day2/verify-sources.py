# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only public source, syntax and include-closure check; no GPU imports."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
EXTERNAL = {
    ("flash-kda/bridge_probe.cu", "kda.cu"),
    ("sparse-native/atlas-wrapper.cu", "kernels/gb10/deepseek-v4-flash/nvfp4/glm_sparse_prefill_kv_reuse.cu"),
    ("sparse-native/atlas-wrapper-reviewed.cu", "kernels/gb10/deepseek-v4-flash/nvfp4/glm_sparse_prefill_kv_reuse.cu"),
    ("moe-route-replay/bench.cu", "capture_routes.hpp"),
}


def main():
    manifest = json.loads((ROOT / "source-map.json").read_text())
    files = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*") if p.is_file()}
    expected = {entry["path"]: entry["public_sha256"] for entry in manifest["files"]}
    assert set(expected) == set(files) - {"source-map.json"}, "source manifest inventory mismatch"
    quoted = 0
    for name, path in files.items():
        if name == "source-map.json":
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[name], name + " hash"
        text = path.read_text()
        if path.suffix == ".py":
            ast.parse(text, filename=name)
        if path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
        if path.suffix in {".cu", ".cuh", ".cpp", ".hpp", ".h"}:
            assert "SPDX-License-Identifier:" in text[:300], name + " license"
            for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', text, re.M):
                quoted += 1
                assert (path.parent / include).is_file() or (name, include) in EXTERNAL, (name, include)
        if path.suffix == ".sha256":
            for line in text.splitlines():
                if not line or line.startswith("#"):
                    continue
                digest, target = line.split(None, 1)
                linked = path.parent / target.strip().lstrip("*")
                assert linked.is_file(), (name, target)
                assert hashlib.sha256(linked.read_bytes()).hexdigest() == digest, (name, target, "hash")
    print(json.dumps({"passed": True, "files": len(expected), "quoted_includes": quoted,
                      "documented_external_includes": len(EXTERNAL), "gpu_run": False}))


if __name__ == "__main__":
    main()
