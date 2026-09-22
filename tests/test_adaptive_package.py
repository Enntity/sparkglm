#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the optional source derivative and its recipe composition."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'research/adaptive-verification'

def module(name):
    spec=importlib.util.spec_from_file_location(name,A/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class PackageTests(unittest.TestCase):
    def fixture(self,root):
        entries=json.loads((A/'source-contract.json').read_text())['files']
        for e in entries:
            dst=root/e['path'];dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(A/'fixtures/installed'/dst.name,dst)
        return entries

    def test_exact_composition_and_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);entries=self.fixture(root);installer=module('install')
            installer.install(root)
            for e in entries:self.assertEqual((root/e['path']).read_bytes(),(A/'receipts'/Path(e['path']).name).read_bytes())
            installer.install(root)

    def test_bad_second_file_leaves_first_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);entries=self.fixture(root)
            first=root/entries[0]['path'];original=first.read_bytes()
            (root/entries[1]['path']).write_text('unqualified source')
            with self.assertRaises(ValueError):module('install').install(root)
            self.assertEqual(first.read_bytes(),original)

    def test_nvme_scheduler_input_is_adaptive_output(self):
        adaptive=json.loads((A/'source-contract.json').read_text())['files'][0]
        nvme=json.loads((ROOT/'overlay/kvoffload/hybrid_patch.json').read_text())['files']
        scheduler=next(e for e in nvme if e['path']==adaptive['path'])
        self.assertEqual(scheduler['base_sha256'],adaptive['result_sha256'])

    def test_renderer_identity_and_entrypoint(self):
        renderer=module('render_profile');one='sha256:'+'1'*64;two='sha256:'+'2'*64
        d=renderer.render(one,two);serialized=json.dumps(d)
        self.assertNotIn('RENDER_REQUIRED',serialized)
        members=d['models'][0]['settings']['placement']['members'];names=[]
        for member in members:
            b=member['runtimeSettings']['bootstrap'];args=b['createArgs']
            self.assertEqual(b['command'],['/opt/sparkglm/adaptive/entrypoint.sh'])
            self.assertFalse(any('dst=/opt/lloom/entrypoint.sh' in x for x in args))
            names.append(next(x for x in args if x.startswith('SPARKGLM_NVME_NAMESPACE=')))
            for kv in ('GLM53_ADAPTIVE_K_SET=2,4,5','MAX_NUM_SEQS=8','SPARKGLM_NVME_CAPACITY_BYTES=68719476736','SPARKGLM_NVME_RESERVE_BYTES=68719476736'):
                self.assertIn(kv,args)
        self.assertEqual(names[0],names[1])
        self.assertNotEqual(d,renderer.render(two,two))
        with self.assertRaises(ValueError):renderer.render('mutable:latest',two)

    def test_entrypoint_preserves_current_hooks_and_capture_geometry(self):
        base=(ROOT/'runtime/entrypoint.sh').read_text()
        expected=base.replace('--cudagraph-capture-sizes 1 2 4 8 16 24 32','--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 12 15 16 18 20 21 24 25 30 32 35 36 40 42 48')
        self.assertEqual(expected,(A/'entrypoint.sh').read_text())
        self.assertIn('--kv-transfer-config',expected)

    def test_retained_upstream_policy_checks(self):
        subprocess.run([sys.executable,str(A/'check_local.py')],check=True,cwd=A)

if __name__=='__main__':unittest.main()
