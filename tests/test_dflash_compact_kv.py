#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Execute the existing draft-group builder, with and without the new overlay."""
import dataclasses
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'overlay'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
compact=load('patch_dflash_compact_kv')
original=load('patch_glm5_drafter_group')
block=original.EDIT_GROUPS_RETURN_NEW
block=block[block.index('    # Drafter group'):block.index('\n    return (')]
SOURCE='def _get_kv_cache_groups_glm5_next(draft_specs, mla_specs, mla_names, mla_page):\n'+block+'\n    return draft_group\n'

@dataclasses.dataclass(frozen=True)
class Spec:
    block_size: int = 16
    page_size_padded: int | None = None
    sliding_window: int = 2048
    bytes_per_token: int = 2048
    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.block_size*self.bytes_per_token

class Uniform:
    @staticmethod
    def from_specs(specs):return specs

def build(source, mode=None, draft=None, mla_block=3584, mla_page=2351104):
    env=dict(replace=dataclasses.replace,UniformTypeKVCacheSpecs=Uniform,
             KVCacheGroupSpec=lambda names,specs:(names,specs),
             logger=SimpleNamespace(info=lambda *a:None))
    exec(source,env)
    with patch.dict(os.environ,{},clear=True):
        if mode is not None:os.environ['SPARKGLM_DFLASH_COMPACT_KV']=mode
        return env['_get_kv_cache_groups_glm5_next'](
            draft if draft is not None else {f'draft.{i}':Spec() for i in range(5)},
            {str(i):SimpleNamespace(block_size=mla_block) for i in range(11)},
            [str(i) for i in range(11)],mla_page)

class Tests(unittest.TestCase):
    def test_disabled_preserves_original_at_multiple_geometries(self):
        changed=compact.patch_text(SOURCE)
        for mode in (None,'0'):
            for block,page in ((3584,2351104),(4096,8388608)):
                self.assertEqual(build(SOURCE,mla_block=block,mla_page=page),
                                 build(changed,mode,mla_block=block,mla_page=page))
        self.assertIsNone(build(changed,'0',draft={}))

    def test_compact_geometry_and_input_immutability(self):
        draft={f'draft.{i}':Spec() for i in range(5)}
        names,specs=build(compact.patch_text(SOURCE),'256',draft)
        self.assertEqual(names,list(draft))
        for spec in specs.values():
            self.assertEqual(spec.block_size,256)
            self.assertIsNone(spec.page_size_padded)
            self.assertEqual(spec.page_size_bytes,524288)
        self.assertTrue(all(s==Spec() for s in draft.values()))
        # 4 contiguous 64-token kernel blocks must exactly fill one manager page.
        self.assertEqual(4*64*Spec().bytes_per_token,524288)

    def test_invalid_flag_and_geometry_rejected(self):
        changed=compact.patch_text(SOURCE)
        for mode in ('','1','128','0256','garbage'):
            with self.assertRaises(ValueError):build(changed,mode)
        for kwargs in (dict(mla_block=4096),dict(mla_page=2351105),
                       dict(draft={str(i):Spec() for i in range(4)}),
                       dict(draft={str(i):Spec(sliding_window=4096) for i in range(5)}),
                       dict(draft={str(i):Spec(bytes_per_token=4096) for i in range(5)}),
                       dict(draft={str(i):Spec(page_size_padded=32768) for i in range(5)})):
            with self.assertRaises(ValueError):build(changed,'256',**kwargs)

    def test_idempotence_and_fail_closed_anchors(self):
        changed=compact.patch_text(SOURCE)
        self.assertEqual(compact.patch_text(changed),changed)
        self.assertEqual(changed.replace(compact.INSERT,''),SOURCE)
        for bad in (SOURCE.replace(compact.ANCHOR,''),SOURCE+compact.ANCHOR,
                    changed.replace('block=256 page=524288','block=128 page=524288'),
                    SOURCE.replace('page_size_padded=mla_page','page_size_padded=0')):
            with self.assertRaises((ValueError,SyntaxError,IndentationError)):
                compact.patch_text(bad)

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cache.py';p.write_text(SOURCE)
            compact.patch_file(str(p),True);self.assertEqual(p.read_text(),SOURCE)
            compact.patch_file(str(p));self.assertEqual(p.read_text(),compact.patch_text(SOURCE))

if __name__=='__main__':unittest.main()
