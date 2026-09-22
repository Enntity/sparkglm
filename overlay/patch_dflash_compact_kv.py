#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Experimental standalone DFlash cache pages; disabled unless explicitly selected.

Apply after patch_glm5_drafter_group.py. SPARKGLM_DFLASH_COMPACT_KV=256
selects only the measured TP2 geometry. Unset or 0 preserves the existing
group builder. This original overlay selects the allocator's existing
standalone tensor path; it changes neither allocation consumers nor kernels.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

DEFAULT_KV_FILE = '/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py'
MARKER = 'SPARKGLM-COMPACT-DRAFT-KV-V1'
ANCHOR = '        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)\n'
INSERT = '''        # SPARKGLM-COMPACT-DRAFT-KV-V1: explicit experimental geometry.
        import os as _compact_kv_os
        _compact_kv_mode = _compact_kv_os.environ.get("SPARKGLM_DFLASH_COMPACT_KV", "0")
        if _compact_kv_mode not in ("0", "256"):
            raise ValueError("SPARKGLM_DFLASH_COMPACT_KV must be 0 or 256")
        if _compact_kv_mode == "256":
            if not (
                mla_block == 3584
                and mla_page == 2351104
                and len(mla_names) == 11
                and len(draft_specs) == 5
                and draft_bytes_per_token == 2048
                and any_draft.page_size_padded is None
                and any_draft.sliding_window == 2048
                and mla_block % 256 == 0
                and 256 * draft_bytes_per_token != mla_page
            ):
                raise ValueError("compact DFlash KV requires the qualified TP2 geometry")
            new_draft_specs = {
                name: replace(s, block_size=256, page_size_padded=None)
                for name, s in draft_specs.items()
            }
            if any(s.page_size_bytes != 524288 for s in new_draft_specs.values()):
                raise ValueError("compact DFlash KV page-size contract changed")
            logger.info("DFlash2 drafter KV: compact standalone block=256 page=524288 layers=5")
'''


def patch_text(text: str) -> str:
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == '_get_kv_cache_groups_glm5_next']
    if len(functions) != 1:
        raise ValueError('expected one GLM5 group builder')
    f = functions[0]
    lines = text.splitlines(keepends=True)
    body = ''.join(lines[f.lineno-1:f.end_lineno])
    if text.count(ANCHOR) != 1 or ANCHOR not in body:
        raise ValueError('draft group anchor changed')
    for required in ('DFLASH2-DRAFTER-GROUP', 'draft_bytes_per_token =',
                     'mla_block =', 'page_size_padded=mla_page'):
        if required not in body:
            raise ValueError('apply padded drafter group patch first: '+required)
    if MARKER in text:
        if text.count(MARKER) != 1 or INSERT + ANCHOR not in body:
            raise ValueError('compact draft patch differs from expected source')
        return text
    result = text.replace(ANCHOR, INSERT + ANCHOR, 1)
    ast.parse(result)
    return result


def patch_file(path: str, dry_run: bool = False) -> int:
    p = Path(path)
    before = p.read_text()
    after = patch_text(before)
    if not dry_run and after != before:
        p.write_text(after)
    print('compact DFlash KV: '+('dry run' if dry_run else 'unchanged' if before == after else 'installed (opt-in)'))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kv-file', default=DEFAULT_KV_FILE)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    return patch_file(args.kv_file, args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
