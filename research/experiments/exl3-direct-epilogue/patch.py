#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Opt-in, hash-locked grouped EXL3 epilogue experiment; never patch the default.

Hadamard arithmetic is adapted unchanged from SparkGLM's ExLlamaV3-derived
helper; M64 foundations retain Reederey provenance. See README.md and the
repository's LICENSES/MIT-ExLlamaV3.txt and LICENSES/NOTICE-Reederey.txt.
Original change: keep each row's transformed values in its owning warp and
write directly to the destination instead of publishing them to shared memory.
"""
import argparse
import hashlib
from pathlib import Path

REFERENCE_SHA256 = 'fee047d2ae88090095bf27916fe0c6fe9ea0c6a29fc8b4e60622a3e16a03f918'
GROUP_START = 'template <bool scatter, bool atomic_scatter = false>\n'
GROUP_END = '__device__ __forceinline__ void grouped_swiglu_had_128('
HELPER_START = '__device__ inline void fat_had_ff_128('
HELPER_END = 'template <int tile_m, bool scatter, bool paired>\n'

DIRECT = '''        // Each warp owns a complete row after the fragment exchange.
        // Read shared scratch once; keep Hadamard results in registers and
        // write them directly. No transformed shared scratch is published.
        for (int row = warp; row < rows; row += 8)
        {
            float4 values = direct_had_ff_128_values(
                sh_c + row * FAT_TILE_N, svh + n_base);
            int source_row = m_base + mb * 16 + row;
            int col_out = lane * 4;
            if constexpr (scatter)
            {
                int64_t destination = token_idx[source_row];
                float weight = __half2float(route_weight[source_row]);
                // Match the reference multiply before the routed accumulation.
                values.x *= weight;
                values.y *= weight;
                values.z *= weight;
                values.w *= weight;
                float* dst = out + destination * out_stride_n + out_n_base + col_out;
                if constexpr (atomic_scatter)
                {
                    atomicAdd(dst + 0, values.x);
                    atomicAdd(dst + 1, values.y);
                    atomicAdd(dst + 2, values.z);
                    atomicAdd(dst + 3, values.w);
                }
                else
                {
                    dst[0] += values.x;
                    dst[1] += values.y;
                    dst[2] += values.z;
                    dst[3] += values.w;
                }
            }
            else
            {
                float* dst = out + source_row * out_stride_n + out_n_base + col_out;
                reinterpret_cast<float4*>(dst)[0] = values;
            }
        }
        // Required before another mb or persistent task reuses sh_c.
        // Rows missing from a partial tile must still participate.
        __syncthreads();
'''


def transform(source: str, coalesced: bool = False) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise ValueError('reference source hash mismatch; review drift, do not force')
    helper_start = source.index(HELPER_START)
    helper_end = source.index(HELPER_END, helper_start)
    helper = source[helper_start:helper_end]
    helper = helper.replace('inline void fat_had_ff_128(', 'inline float4 direct_had_ff_128_values(', 1)
    helper = helper.replace('    float* output_ptr,\n', '', 1)
    helper = helper.replace('    reinterpret_cast<float4*>(output_ptr)[lane] = v;', '    return v;', 1)
    if 'output_ptr' in helper or helper.count('return v;') != 1:
        raise ValueError('unexpected Hadamard helper structure')
    start = source.index(GROUP_START)
    end = source.index(GROUP_END, start)
    group = source[start:end]
    epilogue = group.index('        for (int row = warp; row < rows; row += 8)\n')
    suffix = '    }\n}\n\n'
    if not group.endswith(suffix):
        raise ValueError('unexpected grouped tile boundary')
    # Preserve all fragment math, the producer barrier, and surrounding kernels.
    direct = DIRECT
    if coalesced:
        old = '''                    atomicAdd(dst + 0, values.x);
                    atomicAdd(dst + 1, values.y);
                    atomicAdd(dst + 2, values.z);
                    atomicAdd(dst + 3, values.w);'''
        new = '''                    // Transpose register ownership for contiguous warp atomics.
                    // Every lane participates; branch/row bounds are warp-uniform.
                    #pragma unroll
                    for (int chunk = 0; chunk < 4; ++chunk)
                    {
                        int source_lane = lane / 4 + chunk * 8;
                        float vx = __shfl_sync(0xffffffff, values.x, source_lane);
                        float vy = __shfl_sync(0xffffffff, values.y, source_lane);
                        float vz = __shfl_sync(0xffffffff, values.z, source_lane);
                        float vw = __shfl_sync(0xffffffff, values.w, source_lane);
                        float value = (lane & 3) == 0 ? vx : (lane & 3) == 1 ? vy
                                    : (lane & 3) == 2 ? vz : vw;
                        atomicAdd(dst - col_out + chunk * 32 + lane, value);
                    }'''
        if direct.count(old) != 1:
            raise ValueError('unexpected direct atomic layout')
        direct = direct.replace(old, new)
    updated = group[:epilogue] + direct + suffix
    return source[:start] + helper + updated + source[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--apply', action='store_true', help='modify an isolated build input only')
    parser.add_argument('--coalesced', action='store_true', help='follow-up warp-contiguous atomics experiment')
    args = parser.parse_args()
    source = args.source.read_text()
    candidate = transform(source, coalesced=args.coalesced)
    if args.apply:
        # The Docker path and temporary test copies are allowed; default repo
        # sources are not. This also protects any source checkout, not just ours.
        if args.source.resolve().parent.name == 'overlay':
            parser.error('refusing to modify a reference overlay; use an isolated build input')
        args.source.write_text(candidate)
    print(hashlib.sha256(candidate.encode()).hexdigest())


if __name__ == '__main__':
    main()
