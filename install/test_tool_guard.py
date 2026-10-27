# SPDX-License-Identifier: AGPL-3.0-only
"""Guard defaults and operator configuration through the production launcher."""
import json
from pathlib import Path
import unittest
from test_install import serve

HERE = Path(__file__).resolve().parent


def load_profile(name):
    return json.loads((HERE/'profiles'/f'{name}.json').read_text())


class ToolGuardContract(unittest.TestCase):
    def setUp(self):
        self.env = {'NODE_RANK': '0', 'MASTER_ADDR': '192.0.2.1',
                    'FABRIC_INTERFACE': 'eth0', 'FABRIC_HCA': 'roce0'}

    def test_default_enabled_in_every_mode_rank_and_topology(self):
        cases = [(dict(SPARKGLM_MODE=mode, NODE_RANK=rank), '4x512k')
                 for mode in ('strict', 'spark') for rank in ('0', '1')]
        cases += [(dict(SPARKGLM_STRICT_MTP='1'), '4x1m'),
                  (dict(SPARKGLM_DISPLAY_CARVEOUT='1'), '4x512k'),
                  (dict(FABRIC_HCA='', FABRIC_RIGHT_HCA='roce1'), '4x1m-tp4')]
        for extra, profile in cases:
            with self.subTest(extra=extra, profile=profile):
                args, _ = serve.launch(dict(self.env, **extra), load_profile(profile))
                self.assertIn('--tool-guard=on', args)
                self.assertIn('--tool-guard-max-bytes=1048576', args)

    def test_operator_settings_forwarded_without_shell_interpretation(self):
        args, _ = serve.launch(dict(self.env, SPARKGLM_TOOL_GUARD='off',
            SPARKGLM_TOOL_GUARD_ALLOW_HOSTS='uploads.example.org,api.example.org',
            SPARKGLM_TOOL_GUARD_DENY_HOSTS='blocked.example.org',
            SPARKGLM_TOOL_GUARD_MAX_BYTES='65536'), load_profile('4x512k'))
        self.assertIn('--tool-guard=off', args)
        self.assertIn('--tool-guard-allow-host=uploads.example.org', args)
        self.assertIn('--tool-guard-allow-host=api.example.org', args)
        self.assertIn('--tool-guard-deny-host=blocked.example.org', args)
        self.assertIn('--tool-guard-max-bytes=65536', args)

    def test_invalid_settings_fail(self):
        for key, value in [('SPARKGLM_TOOL_GUARD', 'typo'), ('SPARKGLM_TOOL_GUARD_MAX_BYTES', '0'),
                           ('SPARKGLM_TOOL_GUARD_MAX_BYTES', '67108865'),
                           ('SPARKGLM_TOOL_GUARD_ALLOW_HOSTS', '*.example.org'),
                           ('SPARKGLM_TOOL_GUARD_DENY_HOSTS', 'https://example.org/path')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                serve.launch(dict(self.env, **{key: value}), load_profile('4x512k'))

    def test_profile_cannot_duplicate_guard_flags(self):
        profile = load_profile('4x512k')
        profile['server_argv'].append('--tool-guard=off')
        with self.assertRaises(ValueError):
            serve.launch(self.env, profile)


if __name__ == '__main__':
    unittest.main()
