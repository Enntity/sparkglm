#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Build the Atlas SparkGLM image from the pinned Atlas-Inf/atlas commit
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
version=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$here/release.json")
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
  echo "fetched engine is not the pinned Atlas-Inf/atlas tree $pin_tree" >&2; exit 1; }
git -C "$product" archive HEAD install | tar -x -C "$context"
cp "$here/dockerignore" "$context/.dockerignore"

# Compile jobs: every core, but no more than one per 3 GB of the host memory
# available now, so a build beside running models cannot run a GB10 out of
# memory (it hangs instead of killing a process). ATLAS_BUILD_JOBS overrides.
# Measured on a GB10 (2026-11-24, SparkInkling's engine, a core crate
# touched): release binary and test-suite compile in 239 s at 4 jobs, 132 s
# at 20, with host memory use under 8 GB. The CUDA kernels compile on every
# core whatever this is.
jobs=${ATLAS_BUILD_JOBS:-$(awk -v cores="$(nproc 2>/dev/null || echo 4)" \
  '/^MemAvailable:/ { j = int($2 / 3145728); if (j > cores) j = cores; if (j < 4) j = 4; print j }' \
  /proc/meminfo 2>/dev/null)}
[[ ${jobs:-} =~ ^[1-9][0-9]*$ ]] || jobs=4
args=(--build-arg "INSTALL_TREE=$tree" --build-arg "PRODUCT_VERSION=$version" --build-arg "BUILD_JOBS=$jobs" -f "$context/install/Dockerfile")
echo "building $image with $jobs jobs" >&2
# The release checks (the `checks` stage) run unless ATLAS_IMAGE_CHECKS=0, for
# builds whose engine commit already passed them elsewhere.
case ${ATLAS_IMAGE_CHECKS:-1} in
  1) docker build "${args[@]}" --target checks "$context" >&2 ;;
  0) ;;
  *) echo 'ATLAS_IMAGE_CHECKS must be 0 or 1' >&2; exit 2 ;;
esac
docker build "${args[@]}" --target runtime -t "$image" "$context" >&2
[[ $(docker image inspect -f '{{.Architecture}} {{index .Config.Labels "io.enntity.sparkglm.install-tree"}}' "$image") == "arm64 $tree" ]] || {
  echo "built image $image has the wrong architecture or install-tree label" >&2; exit 1; }
echo "$image"
