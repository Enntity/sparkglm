#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

flash_kda_commit=1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
cutlass_commit=5c149f52a436782210263fb2f19b354443a61c6a
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=${FLASH_KDA_SOURCE:-/tmp/FlashKDA-atlas}
output_dir=${FLASH_KDA_OUTPUT:-$script_dir}

if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/MoonshotAI/FlashKDA.git "$source_dir"
fi
git_source=(git -c "safe.directory=$source_dir" -C "$source_dir")
git_cutlass=(git -c "safe.directory=$source_dir/cutlass" -C "$source_dir/cutlass")
"${git_source[@]}" checkout --detach "$flash_kda_commit"
"${git_source[@]}" submodule update --init cutlass

actual_flash=$("${git_source[@]}" rev-parse HEAD)
actual_cutlass=$("${git_cutlass[@]}" rev-parse HEAD)
[[ "$actual_flash" == "$flash_kda_commit" ]]
[[ "$actual_cutlass" == "$cutlass_commit" ]]

# The patches in order: state slots, TMA swizzle, FP32 recurrence. A tree with
# a prefix of them applied resumes after the last one that reverse-applies.
patches=(
  "$script_dir/flash_kda_sm121_slots.patch"
  "$script_dir/flash_kda_tma_swizzle.patch"
  "$script_dir/flash_kda_fp32_recurrence.patch"
)
applied=0
if ! "${git_source[@]}" diff --quiet; then
  for i in "${!patches[@]}"; do
    if "${git_source[@]}" apply --reverse --check "${patches[$i]}" 2>/dev/null; then
      applied=$((i + 1))
    fi
  done
  if [[ $applied == 0 ]]; then
    echo "FlashKDA source has unrelated changes; refusing to patch" >&2
    exit 2
  fi
fi
for patch in "${patches[@]:$applied}"; do
  "${git_source[@]}" apply "$patch"
done

mkdir -p "$output_dir/build"
common_flags=(
  -std=c++17 -O3 -Xcompiler -fPIC
  -U__CUDA_NO_HALF_OPERATORS__
  -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_HALF2_OPERATORS__
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
  --threads "${NVCC_THREADS:-8}"
  -gencode arch=compute_121f,code=sm_121f
  -I"$source_dir/cutlass/include"
  -I"$source_dir/cutlass/examples/common"
  -I"$source_dir/cutlass/tools/util/include"
  -I"$source_dir/csrc"
  -I"$source_dir/csrc/smxx"
)

nvcc "${common_flags[@]}" \
  -c "$source_dir/csrc/smxx/fwd_launch.cu" \
  -o "$output_dir/build/fwd_launch.o"
nvcc "${common_flags[@]}" \
  -c "$script_dir/atlas_flash_kda_bridge.cu" \
  -o "$output_dir/build/atlas_flash_kda_bridge.o"
nvcc -shared \
  "$output_dir/build/fwd_launch.o" \
  "$output_dir/build/atlas_flash_kda_bridge.o" \
  -o "$output_dir/libatlas_glm53_flash_kda.so"

sha256sum "$output_dir/libatlas_glm53_flash_kda.so"
