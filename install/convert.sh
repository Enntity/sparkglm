#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Convert and verify the checkpoint overlay on THIS node (about a minute, needs
# the GPU). Atlas serves the NVIDIA checkpoint through an overlay that quantizes
# the 864 MTP matrices it needs; everything else links back to the original.
# Skips the work when the overlay was already verified with this converter.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: convert.sh MODEL_ROOT IMAGE' >&2; exit 2; }
model_root=$(realpath "$1") image=$2
src="$model_root/nvidia--GLM-5.3-Flash-NVFP4"
overlay="$model_root/atlas-overlay"
marker="$overlay/sparkglm-verified.json"
[[ -f $src/config.json ]] || { echo "missing checkpoint $src" >&2; exit 2; }
converter=$(docker run --rm --entrypoint sh "$image" -c \
  'cat /opt/atlas/converter/convert.py /opt/atlas/converter/libatlas_mtp_quantize.so | sha256sum' | cut -d' ' -f1)
if [[ -f $marker ]] && grep -q "\"converter_sha256\": \"$converter\"" "$marker"; then
  echo "overlay $overlay already verified"; exit 0
fi
if [[ -e $overlay ]]; then
  aside="$overlay.old-$(date +%Y%m%d-%H%M%S)"
  mv "$overlay" "$aside"
  echo "moved an unverified or older overlay aside to $aside (delete it when satisfied)"
fi
docker run --rm --gpus=all --user "$(id -u):$(id -g)" \
  -v "$src:$src:ro" -v "$model_root:$model_root" \
  --entrypoint python3 "$image" /opt/atlas/converter/convert.py \
  --source "$src" --output "$overlay"
docker run --rm --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$model_root:$model_root" --entrypoint python3 "$image" \
  /opt/atlas/converter/convert.py --source "$src" --output "$overlay" --verify-overlay
printf '{"converter_sha256": "%s", "image": "%s"}\n' "$converter" "$image" > "$marker"
echo "overlay $overlay converted and verified"
