# SPDX-License-Identifier: AGPL-3.0-only
"""CPU checks for the portable managed launch contract."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('atlas_serve', HERE/'serve.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)
DISK_TIER = ('ATLAS_KV_NVME', 'ATLAS_SSM_TIER', 'ATLAS_GLM_NVME')


class LaunchContract(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(serve.profile_path({}, HERE).read_text())
        self.environment = {'NODE_RANK': '0', 'MASTER_ADDR': '192.0.2.1',
                            'FABRIC_INTERFACE': 'enp1s0f0np0', 'MODEL_PATH': '/models/overlay',
                            'DRAFTER_PATH': '/models/drafter'}

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

    def test_profile_matches_the_vllm_context_and_concurrency(self):
        argv, env = serve.launch(self.environment, self.profile)
        # 512K contexts, four sequences over one shared FP8-latent pool.
        self.assertIn('--max-seq-len=524288', argv)
        self.assertIn('--max-num-seqs=4', argv)
        self.assertIn('--kv-cache-dtype=fp8_g128', argv)
        self.assertIn('--gpu-memory-utilization=0.88', argv)
        self.assertIn('--kernel-target=glm-5.3-flash', argv)
        self.assertIn('--ssm-rollback-mode=records', argv)
        self.assertNotIn('ATLAS_KV_OVERCOMMIT', env)
        self.assertEqual(env['ATLAS_GLM_KV_CAP_TO_CONTEXTS'], '0')
        self.assertIn('--oom-guard-mb=4096', argv)
        self.assertIn('--video-max-frames=32', argv)

    def test_dflash_drafter_path_is_substituted_and_validated(self):
        argv, _ = serve.launch(self.environment, self.profile)
        self.assertIn('--dflash', argv)
        self.assertIn('--draft-model=/models/drafter', argv)
        self.assertFalse(any('${' in a for a in argv))
        with self.assertRaises(ValueError):
            serve.launch(dict(self.environment, DRAFTER_PATH='drafter'), self.profile)

    def test_profile_carries_no_diagnostics(self):
        argv, env = serve.launch(self.environment, self.profile)
        self.assertNotIn('--profile', argv)
        for flag in ('ATLAS_DFLASH_STEP_TIMING', 'ATLAS_MTP_TIMING', 'ATLAS_DECODE_BATCH_LOG',
                     'ATLAS_QWEN4EXP_VERIFY_PROF'):
            self.assertNotIn(flag, env)

    def test_ambient_experiments_cannot_change_qualified_profile(self):
        argv, env = serve.launch(dict(self.environment, ATLAS_GLM_DFLASH='0',
                                      ATLAS_GLM_UNREVIEWED_EXPERIMENT='1'), self.profile)
        self.assertEqual(env['ATLAS_GLM_DFLASH'], '1')
        self.assertNotIn('ATLAS_GLM_UNREVIEWED_EXPERIMENT', env)
        self.assertIn('--dflash-gamma=8', argv)

    def test_fabric_hca_lists_and_same_port_siblings(self):
        # GB10 attaches its ConnectX-7 through two PCIe domains, so one cable
        # appears as two RDMA devices (domain 0000 and 0002, same bus:dev.fn).
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pci = root/'pci'
            for dev, addr, state in [('rocep1s0f0', '0000:01:00.0', '4: ACTIVE'),
                                     ('roceP2p1s0f0', '0002:01:00.0', '4: ACTIVE'),
                                     ('rocep1s0f1', '0000:01:00.1', '4: ACTIVE'),
                                     ('roceP2p1s0f1', '0002:01:00.1', '1: DOWN')]:
                (pci/addr).mkdir(parents=True)
                (root/dev/'ports'/'1').mkdir(parents=True)
                (root/dev/'ports'/'1'/'state').write_text(state + '\n')
                (root/dev/'device').symlink_to(pci/addr)
            self.assertEqual(serve.fabric_hcas('rocep1s0f0', root), ['rocep1s0f0', 'roceP2p1s0f0'])
            self.assertEqual(serve.fabric_hcas('rocep1s0f1', root), ['rocep1s0f1'])
            self.assertEqual(serve.fabric_hcas('rocep1s0f0,roceP2p1s0f0', root),
                             ['rocep1s0f0', 'roceP2p1s0f0'])
            self.assertEqual(serve.fabric_hcas('mlx5_0', root/'missing'), ['mlx5_0'])
        argv, env = serve.launch(dict(self.environment, FABRIC_HCA='a0,b1'), self.profile)
        self.assertEqual(env['NCCL_IB_HCA'], 'a0,b1')
        self.assertEqual(env['ATLAS_RDMA_RAILS'], 'a0,b1')
        for bad in ('', 'a0,', 'a0,b1\nX=1', 'a0;b1'):
            with self.subTest(hca=bad), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, FABRIC_HCA=bad), self.profile)

    def test_both_profiles_cache_prompt_prefixes(self):
        # Multi-turn agents resend the whole conversation every turn; without
        # prefix caching each turn re-prefills it from scratch.
        for name in ('4x512k', '8x128k'):
            profile = json.loads(serve.profile_path({'SPARKGLM_PROFILE': name}, HERE).read_text())
            argv, _ = serve.launch(self.environment, profile)
            with self.subTest(profile=name):
                self.assertIn('--enable-prefix-caching', argv)
                slots = [int(a.split('=')[1]) for a in argv if a.startswith('--ssm-cache-slots=')]
                self.assertEqual(len(slots), 1)
                self.assertGreater(slots[0], 0)
                # Only exact, block-aligned prefill states may be restored.
                self.assertEqual(profile['environment'].get('ATLAS_MARCONI_PREFILL_ONLY'), '1')
                # A partial cached block may not be shared: two live sequences
                # would both write the rest of it.
                self.assertEqual(profile['environment'].get('ATLAS_PREFIX_SUBBLOCK'), '0')

    def test_profiles_enable_only_exact_engine_options(self):
        # Every kernel option switched on is bit-exact against the engine with
        # the option off; the lossy and the not-yet-qualified options stay out
        # of a shipped profile.
        envs = [json.loads(serve.profile_path({'SPARKGLM_PROFILE': name}, HERE).read_text())['environment']
                for name in ('4x512k', '8x128k')]
        exact = ('ATLAS_GLM_SPARSE_PREFILL_PIPE', 'ATLAS_GLM_MOE_PREFILL_PERSIST',
                 'ATLAS_GLM_MOE_UNPERMUTE_VEC', 'ATLAS_GLM_INDEX_LOGITS_V2', 'ATLAS_GLM_INDEX_SPLIT',
                 'ATLAS_GLM_ROUTER_PREFILL_CUTLASS', 'ATLAS_GLM_MOE_DECODE_M16',
                 'ATLAS_GLM_MOE_DOWN_ZSKIP', 'ATLAS_GLM_PC_EVICT', 'ATLAS_GLM_PC_BRANCH',
                 'ATLAS_GLM_WARM_SKIP_CACHED', 'ATLAS_GLM_WARM_CHUNK_RUN', 'ATLAS_GLM_CMD_RDMA',
                 'ATLAS_GLM_MOE_DECODE_STREAM', 'ATLAS_GLM_DRAFT_TP', 'ATLAS_GLM_DECODE_FUSE',
                 'ATLAS_GLM_DECODE_GEMV_BATCH', 'ATLAS_GLM_MOE_DECODE_L2PF', 'ATLAS_GLM_MOE_STREAM_NOSYNC',
                 'ATLAS_GLM_DRAFT_TP_BATCH', 'ATLAS_RDMA_ONESHOT', 'ATLAS_RDMA_PAIR_CHAIN')
        # Lossless but not bit-for-bit against the option off: the prefill queue
        # order, and a verify width taken from the drafter's confidence (verify
        # numerics already depend on the width). The first draft of a request no
        # longer reads a row the previous request left behind.
        policy = {'ATLAS_PREFILL_SRPT': '1', 'ATLAS_DFLASH_CONF_WIDTH': '1',
                  'ATLAS_DFLASH_FIRST_APPEND': 'none'}
        held_back = ('ATLAS_GLM_MLA_KVB_MXFP8', 'ATLAS_GLM_INDEX_MXFP8', 'ATLAS_GLM_KV_SHARD',
                     'ATLAS_GLM_PC_FINISH_LEAF', 'ATLAS_GLM_VERIFY_GRAPH', 'ATLAS_GLM_STEP_FUSE',
                     'ATLAS_KV_NVME_DIR', 'ATLAS_GLM_KDA_PREFILL_LT_FP8_SPLITK1',
                     'ATLAS_GLM_TAIL_CUT_DEEP', 'ATLAS_GLM_ZERO_ROWS')
        for env in envs:
            for key in exact:
                self.assertEqual(env.get(key), '1', key)
            for key, value in policy.items():
                self.assertEqual(env.get(key), value, key)
            for key in held_back:
                self.assertNotIn(key, env)

    def test_gpu_memory_utilization_override(self):
        argv, _ = serve.launch(dict(self.environment, SPARKGLM_GPU_MEMORY_UTILIZATION='0.93'), self.profile)
        self.assertIn('--gpu-memory-utilization=0.93', argv)
        self.assertEqual(sum(a.startswith('--gpu-memory-utilization=') for a in argv), 1)
        for bad in ('0.99', '1', '0.5', '0.9 ', '0.93x'):
            with self.subTest(util=bad), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, SPARKGLM_GPU_MEMORY_UTILIZATION=bad), self.profile)

    def test_disk_prefix_cache_is_off_unless_sized(self):
        # Ambient tier variables are dropped like any other ATLAS_* flag.
        for extra in ({}, {'SPARKGLM_PREFIX_CACHE_GB': ''}, {'SPARKGLM_PREFIX_CACHE_GB': '0'},
                      {'ATLAS_KV_NVME_DIR': '/tmp/kv', 'ATLAS_KV_NVME_GB': '24', 'ATLAS_SSM_TIER': '1'}):
            _, env = serve.launch(dict(self.environment, **extra), self.profile)
            with self.subTest(extra=extra):
                self.assertEqual([k for k in env if k.startswith(DISK_TIER)], [])

    def test_disk_prefix_cache_splits_the_size_on_both_ranks(self):
        for rank in ('0', '1'):
            _, env = serve.launch(dict(self.environment, NODE_RANK=rank, SPARKGLM_PREFIX_CACHE_GB='48'),
                                  self.profile)
            self.assertEqual({k: v for k, v in env.items() if k.startswith(DISK_TIER)}, {
                'ATLAS_KV_NVME_DIR': '/prefix-cache/kv', 'ATLAS_KV_NVME_GB': '24',
                'ATLAS_GLM_NVME_FAST': '1', 'ATLAS_SSM_TIER': '1', 'ATLAS_SSM_TIER_UNIFIED': '1',
                'ATLAS_SSM_TIER_SWAP_DIR': '/prefix-cache/ssm', 'ATLAS_SSM_TIER_DISK_GB': '24',
                'ATLAS_SSM_TIER_SLOTS': '2'})
        _, env = serve.launch(dict(self.environment, SPARKGLM_PREFIX_CACHE_GB='17'), self.profile)
        self.assertEqual((env['ATLAS_KV_NVME_GB'], env['ATLAS_SSM_TIER_DISK_GB']), ('8', '9'))
        for bad in ('15', '101', '048', '48.5', '-48', ' 48', '48\n', 'x'):
            with self.subTest(gb=bad), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, SPARKGLM_PREFIX_CACHE_GB=bad), self.profile)

    def test_start_node_mounts_the_disk_prefix_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            models, bin_dir = root/'models', root/'bin'
            for path in ('nvidia--GLM-5.3-Flash-NVFP4/config.json', 'incoai--GLM-5.3-Flash-DFlash2/config.json',
                         'atlas-overlay/sparkglm-verified.json'):
                (models/path).parent.mkdir(parents=True)
                (models/path).touch()
            bin_dir.mkdir()
            # A recording docker; stat and df report ext4 and 100 GiB free unless told otherwise.
            for name, body in (('docker', f'echo "$*" >> {root}/docker.log'),
                               ('stat', 'echo "${FSTYPE:-ext2/ext3}"'),
                               ('df', 'echo "Filesystem 1024-blocks Used Available Capacity Mounted"\n'
                                      'echo "disk 0 0 ${FREE_KB:-104857600} 0 /"')):
                (bin_dir/name).write_text(f'#!/bin/sh\n{body}\n')
                (bin_dir/name).chmod(0o755)

            def start(rank, *extra, **fake):
                (root/'docker.log').unlink(missing_ok=True)
                run = subprocess.run(
                    ['bash', str(HERE/'start-node.sh'), '--rank', rank, '--leader-address', '192.0.2.1',
                     '--model-root', str(models), '--image', 'img', '--fabric-interface', 'enp1s0f0np0',
                     '--cuda-cache', str(root/'cuda'), *extra],
                    capture_output=True, text=True,
                    env=dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', **fake))
                log = (root/'docker.log').read_text() if (root/'docker.log').exists() else ''
                return run, [line for line in log.splitlines() if line.startswith('run ')]

            run, docker_run = start('0')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertNotIn('prefix-cache', docker_run[0].lower())
            cache = root/'nvme'
            for rank in ('0', '1'):
                run, docker_run = start(rank, '--prefix-cache-dir', str(cache))
                with self.subTest(rank=rank):
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertIn(f'--mount type=bind,src={cache.resolve()},dst=/prefix-cache ', docker_run[0])
                    self.assertIn('-e SPARKGLM_PREFIX_CACHE_GB=48 ', docker_run[0])
                    self.assertTrue((cache/'kv').is_dir() and (cache/'ssm').is_dir())
            run, docker_run = start('1', '--prefix-cache-dir', str(cache), '--prefix-cache-gb', '100')
            self.assertIn('-e SPARKGLM_PREFIX_CACHE_GB=100 ', docker_run[0])
            for extra in (['--prefix-cache-dir', 'relative/nvme'], ['--prefix-cache-dir', f'{cache},readonly'],
                          ['--prefix-cache-gb', '48'], ['--prefix-cache-dir', str(cache), '--prefix-cache-gb', '8'],
                          ['--prefix-cache-dir', str(cache), '--prefix-cache-gb', '101']):
                run, docker_run = start('0', *extra)
                with self.subTest(extra=extra):
                    self.assertNotEqual(run.returncode, 0)
                    self.assertEqual(docker_run, [])
                    self.assertIn('prefix-cache', run.stderr)
            for fake in ({'FSTYPE': 'tmpfs'}, {'FSTYPE': 'overlayfs'}, {'FREE_KB': str(47 * 1048576)}):
                run, docker_run = start('0', '--prefix-cache-dir', str(cache), **fake)
                with self.subTest(fake=fake):
                    self.assertNotEqual(run.returncode, 0)
                    self.assertEqual(docker_run, [])
                    self.assertIn('prefix-cache', run.stderr)

    def test_profiles_are_selected_by_name(self):
        self.assertEqual(serve.profile_path({}, HERE), HERE/'profiles'/'4x512k.json')
        small = json.loads(serve.profile_path({'SPARKGLM_PROFILE': '8x128k'}, HERE).read_text())
        argv, _ = serve.launch(self.environment, small)
        self.assertIn('--max-seq-len=131072', argv)
        self.assertIn('--max-num-seqs=8', argv)
        for bad in ('../x', '', 'a/b', '.hidden'):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                serve.profile_path({'SPARKGLM_PROFILE': bad}, HERE)

    def test_invalid_cluster_configuration_fails_before_process_launch(self):
        for key, value in [('NODE_RANK', '2'), ('MASTER_ADDR', 'not-an-address'),
                           ('MASTER_PORT', '0'), ('FABRIC_INTERFACE', 'eth0\nINJECTED=1'),
                           ('MODEL_PATH', 'relative'), ('DRAFTER_PATH', 'rel'),
                           ('SERVED_MODEL_NAME', 'bad\nname')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, **{key:value}), self.profile)


if __name__ == '__main__':
    unittest.main()
