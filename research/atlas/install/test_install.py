# SPDX-License-Identifier: AGPL-3.0-only
"""CPU checks for the portable managed launch contract."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('atlas_serve', HERE/'serve.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class LaunchContract(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((HERE/'profile.json').read_text())
        self.environment = {'NODE_RANK': '0', 'MASTER_ADDR': '192.0.2.1',
                            'FABRIC_INTERFACE': 'enp1s0f0np0', 'MODEL_PATH': '/models/overlay'}

    def test_two_rank_contract_and_private_ports(self):
        for rank in ('0', '1'):
            argv, env = serve.launch(dict(self.environment, NODE_RANK=rank), self.profile)
            self.assertEqual(argv[:3], ['/usr/local/bin/spark', 'serve', '/models/overlay'])
            self.assertIn('--bind=127.0.0.1', argv)
            self.assertIn('--port='+str(8893+int(rank)), argv)
            self.assertIn('--model-name=glm-5.3-flash-atlas', argv)
            self.assertIn('--world-size=2', argv)
            self.assertIn('--video-allow-ffmpeg', argv)
            self.assertIn('--vision-allow-remote-images', argv)
            self.assertNotIn('--vision-remote-image-allow-private', argv)
            self.assertIn('--tp-size=2', argv)
            self.assertIn('--ep-size=2', argv)
            self.assertNotIn('--disable-tool-grammar=true', argv)
            self.assertIn('--tool-call-parser=poolside_v1', argv)
            self.assertEqual(env['NCCL_SOCKET_IFNAME'], self.environment['FABRIC_INTERFACE'])
            self.assertNotIn('NCCL_IB_GID_INDEX', env)

    def test_shared_context_profile_reserves_real_capacity(self):
        argv, env = serve.launch(self.environment, self.profile)
        self.assertIn('--max-seq-len=262144', argv)
        self.assertEqual(env['ATLAS_GLM_SHARED_KV_TOKENS'], '270336')
        self.assertEqual(env['ATLAS_GLM_KV_CAP_TO_CONTEXTS'], '0')
        self.assertEqual(env['ATLAS_KV_ADMIT_WATERMARK'], '262144')
        self.assertEqual(env['ATLAS_KV_OVERCOMMIT'], '0')
        self.assertIn('--oom-guard-mb=4096', argv)
        self.assertIn('--video-max-frames=32', argv)

    def test_ambient_experiments_cannot_change_qualified_profile(self):
        argv, env = serve.launch(dict(self.environment, ATLAS_GLM_MTP_REPAIR='0',
                                      ATLAS_GLM_UNREVIEWED_EXPERIMENT='1'), self.profile)
        self.assertEqual(env['ATLAS_GLM_MTP_REPAIR'], '1')
        self.assertNotIn('ATLAS_GLM_UNREVIEWED_EXPERIMENT', env)
        self.assertIn('--num-drafts=2', argv)

    def test_invalid_cluster_configuration_fails_before_process_launch(self):
        for key, value in [('NODE_RANK', '2'), ('MASTER_ADDR', 'not-an-address'),
                           ('MASTER_PORT', '0'), ('FABRIC_INTERFACE', 'eth0\nINJECTED=1'),
                           ('MODEL_PATH', 'relative'), ('SERVED_MODEL_NAME', 'bad\nname')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, **{key:value}), self.profile)


if __name__ == '__main__':
    unittest.main()
