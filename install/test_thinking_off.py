# SPDX-License-Identifier: AGPL-3.0-only
"""A request that turns thinking off: capped Low effort in Spark mode, the
reference server's behavior in Strict mode, and an operator may choose either."""
import unittest
from test_install import serve
from test_tool_guard import load_profile

FLAG = '--thinking-off-budget=256'


class ThinkingOffContract(unittest.TestCase):
    def setUp(self):
        self.env = {'NODE_RANK': '0', 'MASTER_ADDR': '192.0.2.1',
                    'FABRIC_INTERFACE': 'eth0', 'FABRIC_HCA': 'roce0'}

    def args(self, **extra):
        return serve.launch(dict(self.env, **extra), load_profile('4x512k'))[0]

    def test_mode_decides_the_default_on_every_rank(self):
        for rank in ('0', '1'):
            with self.subTest(rank=rank):
                self.assertIn(FLAG, self.args(SPARKGLM_MODE='spark', NODE_RANK=rank))
                self.assertNotIn(FLAG, self.args(SPARKGLM_MODE='strict', NODE_RANK=rank))

    def test_operator_choice_wins_in_either_mode(self):
        self.assertNotIn(FLAG, self.args(SPARKGLM_MODE='spark', SPARKGLM_THINKING_OFF='reference'))
        self.assertIn(FLAG, self.args(SPARKGLM_MODE='strict', SPARKGLM_THINKING_OFF='low'))

    def test_invalid_or_duplicated_setting_fails(self):
        with self.assertRaises(ValueError):
            self.args(SPARKGLM_THINKING_OFF='off')
        profile = load_profile('4x512k')
        profile['server_argv'].append('--thinking-off-budget=64')
        with self.assertRaises(ValueError):
            serve.launch(self.env, profile)


if __name__ == '__main__':
    unittest.main()
