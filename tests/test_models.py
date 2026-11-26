# SPDX-License-Identifier: AGPL-3.0-only
"""--model (or SPARKGLM_MODEL) picks the pinned checkpoint ./start.sh serves."""
import pathlib
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'install'))
import test_fabric  # noqa: E402  (its start.sh harness: fake hosts, Docker and converter)

GLM = 'nvidia/GLM-5.3-Flash-FP4-Local'
SAGE = 'Enntity/Sage-S2.1-Flash-NVFP4'


class Models(unittest.TestCase):
    def start(self, env_text, repo, *args):
        real = subprocess.run

        def run(cmd, *a, **k):
            if cmd[:1] == ['bash'] and cmd[1].endswith('/start.sh'):
                cmd = [*cmd, *args]
            return real(cmd, *a, **k)
        with mock.patch.object(test_fabric.subprocess, 'run', run):
            return test_fabric.CheckpointOverride.run_start(self, env_text, repo)

    def assert_serves(self, result, calls, converts, repo):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(converts, [])
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertRegex(call, rf'MODEL_PATH=\S*/models/{repo.replace("/", "--")} ')

    def test_default_is_glm(self):
        self.assert_serves(*self.start('WORKER=w\n', GLM), GLM)
        self.assert_serves(*self.start('WORKER=w\n', GLM, '--model', 'glm-5.3-flash'), GLM)

    def test_sage_by_flag_or_env_in_either_order_with_mode(self):
        for env, args in [('WORKER=w\n', ('--model', 'sage-s2.1-flash')),
                          ('WORKER=w\n', ('--mode', 'strict', '--model', 'sage-s2.1-flash')),
                          ('WORKER=w\n', ('--model', 'sage-s2.1-flash', '--mode', 'strict')),
                          ('WORKER=w\nSPARKGLM_MODEL=sage-s2.1-flash\n', ())]:
            with self.subTest(env=env, args=args):
                result, calls, converts = self.start(env, SAGE, *args)
                self.assert_serves(result, calls, converts, SAGE)
                if 'strict' in args:
                    self.assertIn('SPARKGLM_MODE=strict', calls[0])

    def test_unknown_model_is_refused_before_anything_runs(self):
        result, calls, converts = self.start('WORKER=w\n', GLM, '--model', 'sage')
        self.assertEqual(result.returncode, 2)
        self.assertIn('glm-5.3-flash or sage-s2.1-flash', result.stderr)
        self.assertEqual((calls, converts), ([], []))

    def test_checkpoint_override_is_glm_only(self):
        env = ('WORKER=w\nCHECKPOINT_REPO=nvidia/GLM-5.3-Flash-NVFP4\n'
               'CHECKPOINT_REVISION=423acf37583782c51c142d145aef733d72943d93\n')
        result, calls, converts = self.start(env, SAGE, '--model', 'sage-s2.1-flash')
        self.assertEqual(result.returncode, 2)
        self.assertIn('CHECKPOINT_REPO', result.stderr)
        self.assertEqual((calls, converts), ([], []))


if __name__ == '__main__':
    unittest.main()
