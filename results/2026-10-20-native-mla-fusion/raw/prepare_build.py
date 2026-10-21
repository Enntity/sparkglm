# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
SRC, PRODUCT, CTX = ROOT / "engine", ROOT / "product", ROOT / "context"
CTX.mkdir()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


assert not git(SRC, "status", "--porcelain")
assert not git(PRODUCT, "status", "--porcelain")
old = json.loads((ROOT.parent / "glm-native-mtp-serving-20261020/context.attempt3/source-hashes.json").read_text())
tracked = git(SRC, "ls-files").splitlines()
roots = ("crates/", "kernels/", "vendor/", "jinja-templates/", "test_data/", "bench/ngram_ref/")
hashes = {f: hashlib.sha256((SRC / f).read_bytes()).hexdigest()
          for f in tracked if (f.startswith(roots) or f in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"))
          and (SRC / f).is_file()}
changed = []
for f, digest in hashes.items():
    if old.get(f) != digest:
        dest = CTX / "patch" / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / f, dest)
        changed.append(f)
(CTX / "source-hashes.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")
shutil.copy2(ROOT.parent / "glm-tp2-linear-width-20261020/red-context/verify_source.py", CTX)
shutil.copytree(PRODUCT / "install", CTX / "install", ignore=shutil.ignore_patterns("__pycache__"))
commit, tree = git(SRC, "rev-parse", "HEAD"), git(SRC, "rev-parse", "HEAD^{tree}")
install_tree = git(PRODUCT, "rev-parse", "HEAD:install")
tag = "localhost/atlas-sparkglm:native-mla-fuse-" + install_tree[:12]
(CTX / "Dockerfile").write_text(f"""# SPDX-License-Identifier: AGPL-3.0-only
FROM localhost/atlas-native-mtp-builder-v5:20261020 AS builder
WORKDIR /build
COPY patch/ /build/
COPY source-hashes.json verify_source.py /tmp/
RUN python3 /tmp/verify_source.py && cargo test --locked --release -j 4 -p spark-model --lib glm_mla_fuse && cargo build --locked --release -j 4 -p spark-server
FROM localhost/atlas-sparkglm:tp2-modes-516a51195996
COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY install/serve.py /opt/atlas/serve.py
COPY install/profiles/ /opt/atlas/profiles/
COPY install/atlas-source.json /opt/atlas/source-manifest.json
LABEL io.enntity.sparkglm.install-tree="{install_tree}" \\
      io.enntity.sparkglm.engine-commit="{commit}" \\
      io.enntity.sparkglm.local-rc="true"
""")
with tarfile.open(ROOT / "context.tar", "w") as tf:
    for p in CTX.iterdir():
        tf.add(p, arcname=p.name)
plan = {"commit": commit, "tree": tree, "product_commit": git(PRODUCT, "rev-parse", "HEAD"),
        "install_tree": install_tree, "image": tag, "changed_source": changed,
        "verified_source_files": len(hashes),
        "context_sha256": hashlib.sha256((ROOT / "context.tar").read_bytes()).hexdigest(),
        "builder_image": "sha256:adc3afbe66ce84f3d0ded8d7734dc912a31491dc92a7b99f1fa525595c637600",
        "runtime_base": "sha256:3de7af108f595066b5e05b1dd5c171ebbc534e9dd5c6cb80f2e58087e5e44d3c"}
(ROOT / "build-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print(f"Prepared {len(changed)} changed source files; verify all {len(hashes)} source hashes.")
