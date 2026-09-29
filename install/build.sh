#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the Atlas SparkGLM image from the pinned Enntity/atlas commit
# (atlas-source.json) and this directory. The image tag is the git tree hash of
# install/, so a published image and a local build of the same checkout carry
# the same tag. Never starts, stops or converts anything.
#
#   build.sh         build the image, then print its tag
#   build.sh --tag   print the tag only
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
product=$(cd "$here/.." && pwd)
tree=$(git -C "$product" rev-parse HEAD:install)
image=${SPARKGLM_IMAGE_REPO:-ghcr.io/enntity/atlas-sparkglm}:${tree:0:12}
[[ ${1:-} == --tag ]] && { echo "$image"; exit 0; }
[[ $# == 0 ]] || { echo 'usage: build.sh [--tag]' >&2; exit 2; }
[[ -z $(git -C "$product" status --porcelain --untracked-files=all -- install) ]] || {
  echo 'install/ has local changes; commit them first (the image tag is its git tree)' >&2; exit 2; }
read -r repo commit pin_tree < <(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["repository"], m["commit"], m["tree"])' "$here/atlas-source.json")

mkdir -p "${SPARKGLM_BUILD_DIR:=$HOME/.cache/sparkglm}"
context=$(mktemp -d "$SPARKGLM_BUILD_DIR/build.XXXXXX")
trap 'rm -rf "$context"' EXIT
# A shallow fetch of the pinned commit; Git LFS media (demo assets) is not needed.
git init -q "$context/engine"
GIT_LFS_SKIP_SMUDGE=1 git -C "$context/engine" fetch -q --depth 1 "$repo" "$commit"
GIT_LFS_SKIP_SMUDGE=1 git -C "$context/engine" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
[[ $(git -C "$context/engine" rev-parse HEAD) == "$commit" &&
   $(git -C "$context/engine" rev-parse 'HEAD^{tree}') == "$pin_tree" ]] || {
  echo "fetched engine is not the pinned Enntity/atlas tree $pin_tree" >&2; exit 1; }
git -C "$product" archive HEAD install | tar -x -C "$context"
cp "$here/dockerignore" "$context/.dockerignore"

docker build --build-arg "INSTALL_TREE=$tree" \
  --build-arg "BUILD_JOBS=${ATLAS_BUILD_JOBS:-4}" \
  -f "$context/install/Dockerfile" -t "$image" "$context" >&2
[[ $(docker image inspect -f '{{.Architecture}} {{index .Config.Labels "io.enntity.sparkglm.install-tree"}}' "$image") == "arm64 $tree" ]] || {
  echo "built image $image has the wrong architecture or install-tree label" >&2; exit 1; }
echo "$image"
