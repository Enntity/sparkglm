# SPDX-License-Identifier: AGPL-3.0-only
"""Four-rank modes and physical topology must resolve to one launch policy."""
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import test_install

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('tp4_serve', HERE / 'serve.py')
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)


class FourRankModes(unittest.TestCase):
    def launch(self, rank=0, topology='chain', mode='strict', mtp=False):
        profile = json.loads((HERE / 'profiles/4x1m-tp4.json').read_text())
        env = dict(NODE_RANK=str(rank), MASTER_ADDR='192.0.2.1',
                   FABRIC_INTERFACE='fabric0', FABRIC_SIBLINGS='0',
                   SPARKGLM_MODE=mode, SPARKGLM_TOPOLOGY=topology,
                   MODEL_PATH='/models/checkpoint', DRAFTER_PATH='/models/drafter',
                   SPARKGLM_PREFIX_CACHE_GB='48', SPARKGLM_KV_SHARD='1')
        if rank > 0 or topology == 'ring':
            env['FABRIC_LEFT_HCA'] = 'left0,left1'
        if rank < 3 or topology == 'ring':
            env['FABRIC_RIGHT_HCA'] = 'right0,right1'
        if mtp:
            env['SPARKGLM_STRICT_MTP'] = '1'
        return serve.launch(env, profile)

    def test_both_modes_on_every_rank_and_topology(self):
        for topology in ('chain', 'ring'):
            for mode in ('strict', 'spark'):
                for rank in range(4):
                    with self.subTest(topology=topology, mode=mode, rank=rank):
                        args, env = self.launch(rank, topology, mode)
                        self.assertIn('--world-size=4', args)
                        self.assertEqual(env['ATLAS_GLM_SERVING_MODE'], mode)
                        self.assertEqual(env['ATLAS_RDMA_TOPOLOGY'], topology)
                        self.assertEqual(env['ATLAS_GLM_FP4L_KDA_MXFP8_DECODE'],
                                         '1' if mode == 'strict' else '0')
                        self.assertEqual(env['ATLAS_GLM_FP4L_DECODE_A16'],
                                         '1' if mode == 'strict' else '0')
                        # TP4 Strict also reproduces the arithmetic miss when
                        # the shared expert is split; both release modes use
                        # its full output before blending.
                        self.assertEqual(env['ATLAS_GLM_SHARED_TP_SPLIT'], '0')
                        self.assertIn(f'/tp4/{topology}/{mode}/', env['ATLAS_KV_NVME_DIR'])

    def test_native_mtp_preserves_capacity_without_copying_pair_memory_cap(self):
        for topology in ('chain', 'ring'):
            args, env = self.launch(topology=topology, mtp=True)
            self.assertIn('--speculative', args)
            self.assertNotIn('--dflash', args)
            self.assertIn('--max-num-seqs=4', args)
            self.assertIn('--max-seq-len=1048576', args)
            self.assertIn('--enable-prefix-caching', args)
            self.assertEqual(env['ATLAS_GLM_STRICT_NATIVE_MTP'], '1')
            self.assertEqual(env['ATLAS_GLM_LONG_BATCH_VERIFY'], '0')
            self.assertEqual(env['ATLAS_LARGE_ENCODE_CONCURRENCY'], '2')
            self.assertEqual(env['ATLAS_TOKENIZER_IDS_ONLY'], '1')
            self.assertNotIn('ATLAS_KV_MAX_BLOCKS', env)

    def test_ring_requires_wraparound_rails_and_rejects_ambiguous_links(self):
        for rank in range(4):
            env = dict(SPARKGLM_TOPOLOGY='ring', FABRIC_LEFT_HCA='left',
                       FABRIC_RIGHT_HCA='right')
            rails = serve.fabric_rails(env, rank, 4, False)
            self.assertEqual(rails['ATLAS_RDMA_LEFT_RAILS'], 'left')
            self.assertEqual(rails['ATLAS_RDMA_RIGHT_RAILS'], 'right')
            for missing in ('FABRIC_LEFT_HCA', 'FABRIC_RIGHT_HCA'):
                with self.assertRaises(ValueError):
                    serve.fabric_rails({k: v for k, v in env.items() if k != missing}, rank, 4, False)
            with self.assertRaises(ValueError):
                serve.fabric_rails(dict(env, FABRIC_LEFT_HCA='right'), rank, 4, False)

    def test_topology_does_not_silently_fall_back(self):
        for value in ('', 'mesh', 'RING', 'pair'):
            with self.assertRaises(ValueError):
                self.launch(topology=value)
        with self.assertRaises(ValueError):
            serve.fabric_rails(dict(SPARKGLM_TOPOLOGY='ring', FABRIC_HCA='rail'), 0, 2, False)


class FourRankNodeLaunch(unittest.TestCase):
    def test_each_ring_rank_passes_mode_and_wraparound_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, start = test_install.LaunchContract.fake_node(Path(tmp))
            for rank in range(4):
                for mode in ('strict', 'spark'):
                    result, calls = start(str(rank), '--world-size', '4', '--topology', 'ring',
                                          '--mode', mode, '--left-hca', 'rocep1s0f0',
                                          '--right-hca', 'rocep1s0f1')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('-e SPARKGLM_TOPOLOGY=ring ', calls[0])
                    self.assertIn('-e SPARKGLM_PROFILE=4x1m-tp4 ', calls[0])
                    self.assertIn(f'-e SPARKGLM_MODE={mode} ', calls[0])
            for extra in (['--world-size', '2', '--topology', 'ring'],
                          ['--world-size', '4', '--topology', 'ring', '--right-hca', 'rocep1s0f1'],
                          ['--world-size', '4', '--topology', 'chain', '--left-hca', 'rocep1s0f0', '--right-hca', 'rocep1s0f1'],
                          ['--strict-mtp', '--mode', 'spark']):
                result, calls = start('0', *extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
