#!/usr/bin/env python3
"""Local-only policy/patch checks. No serving imports, network, SSH, or GPU."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'vendor/mia/patch_adaptive_k.py'
CONSTANTS = {}
for node in ast.parse(UPSTREAM.read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
        for target in node.targets:
            if isinstance(target, ast.Name):
                CONSTANTS[target.id] = node.value.value


class Request:
    def __init__(self, name, structured=False, prefill=False):
        self.request_id = name
        self.use_structured_output = structured
        self.is_prefill_chunk = prefill
        self.spec_token_ids = list(range(5))


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {
            'GLM53_ADAPTIVE_K': 'ema', 'GLM53_ADAPTIVE_K_SET': '2,4,5',
            'GLM53_ADAPTIVE_K_MIN_STEPS': '4', 'GLM53_ADAPTIVE_K_ALPHA': '0.25',
            'GLM53_ADAPTIVE_K_MARGIN': '1.0', 'GLM53_ADAPTIVE_K_SATURATE': 'max',
            'GLM53_ADAPTIVE_K_HIST': '200',
            'GLM53_ADAPTIVE_K_FILE': str(Path(self.temp.name) / 'unused.json'),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        namespace = {'os': os}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(CONSTANTS['SCHED_HELPER'], namespace)
        self.policy = namespace['_GLM53_ADAPTIVE_K']

    def observe_low(self, request):
        for _ in range(20):
            self.policy.observe(request.request_id, 5, 1)

    def test_cold_and_structured_batch_guards(self):
        a, b, s = Request('a'), Request('new'), Request('structured', True)
        self.observe_low(a)
        live = {x.request_id: x for x in [a, b, s]}
        self.assertEqual(self.policy.batch_k(5, [a], live), 2)
        self.assertEqual(self.policy.batch_k(5, [a, b], live), 5)
        self.assertEqual(self.policy.batch_k(5, [a, s], live), 5)

    def test_warmup_and_recovery(self):
        a = Request('a')
        for _ in range(3):
            self.policy.observe('a', 5, 0)
            self.assertEqual(self.policy.batch_k(5, [a], {'a': a}), 5)
        self.observe_low(a)
        self.assertEqual(self.policy.batch_k(5, [a], {'a': a}), 2)
        for _ in range(25):
            k = self.policy.batch_k(5, [a], {'a': a})
            self.policy.observe('a', k, k)
        self.assertEqual(self.policy.batch_k(5, [a], {'a': a}), 5)

    def test_uniform_sync_prefix_and_async_lengths(self):
        a, b = Request('a'), Request('b')
        self.observe_low(a)
        for _ in range(20): self.policy.observe('b', 5, 5)
        self.policy.apply([(a, False), (b, False)], {'a': a, 'b': b})
        self.assertEqual(a.spec_token_ids, [0, 1])
        self.assertEqual(b.spec_token_ids, [0, 1])
        self.assertEqual(self.policy.batch_k(5, [a, b], {'a': a, 'b': b}), 2)

    def test_disabled_policy(self):
        a = Request('a')
        self.observe_low(a)
        self.policy.enabled = False
        self.assertEqual(self.policy.batch_k(5, [a], {'a': a}), 5)
        self.policy.apply([(a, False)], {'a': a})
        self.assertEqual(a.spec_token_ids, list(range(5)))

    def test_graph_geometry_c1_through_c8(self):
        namespace = {}
        exec(CONSTANTS['CG_HELPER'], namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            lengths = namespace['_glm53_adaptive_k_query_lens']([6], 6)
        self.assertEqual(lengths, [3, 5, 6])
        manifest = json.loads((ROOT / 'experiment.json').read_text())
        captures = set(manifest['captureSizesBothArms'])
        for n in range(1, 9):
            for length in lengths: self.assertIn(n * length, captures)

    def test_patch_compatibility_idempotence_and_drift_rejection(self):
        paths = {}
        for name in ['scheduler.py', 'cudagraph_utils.py']:
            dest = Path(self.temp.name) / name
            dest.write_bytes((ROOT / 'fixtures/vllm' / name).read_bytes())
            paths[name] = dest
        env = dict(os.environ, GLM53_SCHEDULER_PY=str(paths['scheduler.py']),
                   GLM53_CUDAGRAPH_UTILS_PY=str(paths['cudagraph_utils.py']))
        argv = [sys.executable, '-S', str(UPSTREAM)]
        run = subprocess.run(argv, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        first = {name: path.read_text() for name, path in paths.items()}
        for name, source in first.items(): compile(source, name, 'exec')
        run = subprocess.run(argv, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(first, {name: path.read_text() for name, path in paths.items()})
        paths['scheduler.py'].write_text('import time\n')
        run = subprocess.run(argv, env=env, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
