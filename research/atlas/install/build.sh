#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Source installer. Never changes model residency or starts an inference server.
set -euo pipefail
[[ $# == 1 ]] || { echo 'usage: build.sh INSTALL_ROOT' >&2; exit 2; }
product=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
install_root=$1
mkdir -p "$install_root"
install_root=$(realpath "$install_root")
source_revision=$(git -C "$product" rev-parse HEAD)
[[ -z $(git -C "$product" status --porcelain --untracked-files=all) ]] || {
  echo 'Build from a clean, pinned SparkGLM checkout' >&2; exit 2;
}
manifest="$product/research/atlas/nvidia-installable-source.json"
[[ -f "$manifest" ]] || { echo 'Installable engine source manifest is missing' >&2; exit 2; }
image="lloom/atlas-sparkglm:$source_revision"
context="$install_root/build-$source_revision"
receipt="$install_root/image-$source_revision.json"
if [[ -f "$receipt" ]]; then
  python3 - "$receipt" "$image" "$source_revision" "$manifest" <<'PY'
import hashlib, json, pathlib, subprocess, sys
receipt = json.load(open(sys.argv[1]))
image, revision, manifest = sys.argv[2:]
actual = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
expected_manifest = hashlib.sha256(pathlib.Path(manifest).read_bytes()).hexdigest()
if (receipt['image'] != image or actual['Id'] != receipt['image_id']
        or receipt['source_revision'] != revision
        or receipt['manifest_sha256'] != expected_manifest
        or actual['Architecture'] != 'arm64'
        or actual['Config'].get('Labels', {}).get('org.opencontainers.image.revision') != revision):
    raise SystemExit('Installed image identity changed; refusing to reuse it')
print(actual['Id'])
PY
  exit 0
fi
upstream="$install_root/atlas-upstream"
if [[ ! -d "$upstream/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/Mango-kid/atlas.git "$upstream"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$upstream" checkout --detach 90b3584abc71b44b609637092b85d8423d8ff20f
fi
if [[ ! -d "$context" ]]; then
  mkdir "$context"
  python3 "$product/research/atlas/project/reconstruct.py" --checkpoint installable \
    --source-repo "$upstream" --destination "$context/engine" > "$context/reconstruction.json"
  mkdir -p "$context/product/research/atlas"
  git -C "$product" archive HEAD research/atlas/install research/atlas/flash_kda \
    research/atlas/nvidia-mtp-converter | tar -x -C "$context/product"
  cp "$manifest" "$context/source-manifest.json"
  cp "$product/research/atlas/install/dockerignore" "$context/.dockerignore"
fi
# Reused contexts must match the source manifest; failed preparation is retained
# for inspection rather than silently being mistaken for a successful checkout.
python3 - "$manifest" "$context" "$product" <<'PY'
import hashlib, json, pathlib, subprocess, sys
manifest, context, product = map(pathlib.Path, sys.argv[1:])
expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
receipt = json.loads((context/'reconstruction.json').read_text())
if receipt['manifest_sha256'] != expected or (context/'source-manifest.json').read_bytes() != manifest.read_bytes():
    raise SystemExit('Build context source receipt mismatch')
if (context/'.dockerignore').read_bytes() != (product/'research/atlas/install/dockerignore').read_bytes():
    raise SystemExit('Build context ignore rules changed')
engine = context/'engine'
def git(*args):
    return subprocess.check_output(['git', '-C', str(engine), *args], text=True).strip()
if (git('write-tree') != json.loads(manifest.read_text())['reconstructed_tree']
        or git('diff', '--name-only')
        or git('ls-files', '--others')):
    raise SystemExit('Reconstructed engine was modified; refusing build context reuse')
def files(root):
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise SystemExit('Install asset symlinks are prohibited')
        if path.is_file():
            result[str(path.relative_to(root))] = (path.read_bytes(), path.stat().st_mode & 0o111)
    return result
for name in ('install', 'flash_kda', 'nvidia-mtp-converter'):
    relative = pathlib.Path('research/atlas')/name
    names = subprocess.check_output(['git', '-C', str(product), 'ls-files', '-z', str(relative)], text=True).split('\0')
    original = {str(pathlib.Path(name).relative_to(relative)): ((product/name).read_bytes(), (product/name).stat().st_mode & 0o111) for name in names if name}
    if original != files(context/'product'/relative):
        raise SystemExit('Packaged install assets were modified: '+name)

PY
docker build --build-arg "SPARKGLM_REVISION=$source_revision" \
  --build-arg "BUILD_JOBS=${ATLAS_BUILD_JOBS:-4}" \
  -f "$context/product/research/atlas/install/Dockerfile" -t "$image" "$context"
python3 - "$receipt" "$image" "$source_revision" "$manifest" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
target, image, revision, manifest = sys.argv[1:]
identity = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
if identity['Architecture'] != 'arm64' or identity['Config']['Labels']['org.opencontainers.image.revision'] != revision:
    raise SystemExit('Built image architecture/revision mismatch')
data = dict(image=image, image_id=identity['Id'], source_revision=revision,
            manifest_sha256=hashlib.sha256(pathlib.Path(manifest).read_bytes()).hexdigest())
tmp = pathlib.Path(target+'.tmp')
tmp.write_text(json.dumps(data, indent=2)+'\n')
os.replace(tmp, target)
print(json.dumps(data))
PY
