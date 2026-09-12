#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Mechanical partition/store adapter; asserts original QK/softmax/PV body intact."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
source = P/'production/kernels/gb10/deepseek-v4-flash/nvfp4/glm_sparse_prefill_kv_reuse.cu'
original = source.read_text()
text = original.replace('glm_kvp_', 'glm_split_kvp_')


def replace(old, new):
    global text
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)


replace('glm_sparse_mla_prefill_bf16_head32_tc_kv_pad(', 'atlas_sparse_decode_split(')
replace('    __nv_bfloat16* O,', '    float* O,')
replace('    float inv_sqrt_d\n', '    float inv_sqrt_d,\n    float* LSE, unsigned int splits\n')
replace('    const unsigned int token_row = blockIdx.y;',
        '    if (splits < 1 || splits > 16 || blockIdx.z >= splits) return;\n'
        '    const unsigned int token_row = blockIdx.y;')
replace('    O += ((unsigned long long)token_row * num_heads + head_start) * head_dim;',
        '    O += (((unsigned long long)blockIdx.z * rows + token_row) * num_heads + head_start) * head_dim;\n'
        '    LSE += ((unsigned long long)blockIdx.z * rows + token_row) * num_heads + head_start;')
replace('    unsigned int num_kv_blocks = (kv_len + BC_512 - 1) / BC_512;',
        '    unsigned int num_kv_blocks = (kv_len + BC_512 - 1) / BC_512;\n'
        '    const unsigned int first_block = num_kv_blocks * blockIdx.z / splits;\n'
        '    const unsigned int end_block = num_kv_blocks * (blockIdx.z + 1) / splits;')
replace('        if (num_kv_blocks > 0) {', '        if (first_block < end_block) {')
replace('LOAD_KV_TILE_512(K_cache, block_table, smem_K, 0, kv_len, tid, blockDim.x);',
        'LOAD_KV_TILE_512(K_cache, block_table, smem_K, first_block*BC_512, kv_len, tid, blockDim.x);')
replace('for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++)',
        'for (unsigned int kv_block = first_block; kv_block < end_block; kv_block++)')
replace('if(kv_block+1 < num_kv_blocks)', 'if(kv_block+1 < end_block)')
replace('        __nv_bfloat16* ob = O;',
        '        if (warp_id < 2 && tid_in_group == 0) {\n'
        '            if (r0 < q_len) LSE[r0] = l_r0 > 0 ? m_r0 + logf(l_r0) : -INFINITY;\n'
        '            if (r1 < q_len) LSE[r1] = l_r1 > 0 ? m_r1 + logf(l_r1) : -INFINITY;\n'
        '        }\n'
        '        float* ob = O;')
for row, left, right, inv in [('gr0', 0, 1, 'il0'), ('gr1', 2, 3, 'il1')]:
    replace(f'                unsigned int lo=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][{left}]*{inv}));\n'
            f'                unsigned int hi=(unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][{right}]*{inv}));\n'
            f'                *(unsigned int*)&ob[{row}*q_seq_stride+c0]=lo|(hi<<16);',
            f'                ob[{row}*q_seq_stride+c0] = acc_o[nt][{left}]*{inv};\n'
            f'                ob[{row}*q_seq_stride+c0+1] = acc_o[nt][{right}]*{inv};')
begin = '        unsigned int kv_start = kv_block * BC_512;'
end = '        // === Sequential K[next] load'
assert original[original.index(begin):original.index(end)] == text[text.index(begin):text.index(end)].replace('glm_split_kvp_', 'glm_kvp_')
(P/'generated.cuh').write_text(text)
(P/'generation.json').write_text(json.dumps(dict(
    baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(text.encode()).hexdigest(),
    original_qk_softmax_pv_body_unchanged=True,
    changes=['helper symbol namespace', 'tile interval partition', 'FP32 partial output', 'natural LSE store']), indent=2)+'\n')
print('PASS original QK/softmax/PV body unchanged; generated split kernel')
