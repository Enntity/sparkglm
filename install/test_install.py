# SPDX-License-Identifier: AGPL-3.0-only
"""CPU checks for the portable managed launch contract."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('atlas_serve', HERE/'serve.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)
DISK_TIER = ('ATLAS_KV_NVME', 'ATLAS_SSM_TIER', 'ATLAS_GLM_NVME')
# The engine's refusals at world > 2 (serve_phases/preflight/glm_chain.rs).
REFUSED_BEYOND_TWO_RANKS = (
    'ATLAS_GLM_SPARSE_NATIVE', 'ATLAS_GLM_K3_MLA_QUERY_COMPARE', 'ATLAS_GLM_KV_SHARD',
    'ATLAS_GLM_C4_DECODE', 'ATLAS_GLM_C4_SPARSE', 'ATLAS_GLM_C4_GROUPED_MOE', 'ATLAS_GLM_INDEPENDENT_DECODE',
    'ATLAS_GLM_MTP_REPAIR', 'ATLAS_GLM_MTP_DISTRIBUTED')


class LaunchContract(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(serve.profile_path({}, HERE).read_text())
        self.environment = {'NODE_RANK': '0', 'MASTER_ADDR': '192.0.2.1',
                            'FABRIC_INTERFACE': 'enp1s0f0np0', 'FABRIC_HCA': 'rocep1s0f0', 'MODEL_PATH': '/models/overlay',
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

    def test_mtp_layer_comes_quantized_or_through_the_overlay(self):
        # FP4-Local ships its MTP experts (layer 45) as W4A16 NVFP4 and is served
        # as published; the NVFP4 release's BF16 ones need convert.sh's receipt.
        experts = 'model.language_model.layers.{}.mlp.experts'
        fp4l = {'text_config': {'num_hidden_layers': 45},
                'quantization_config': {'quant_method': 'modelopt', 'quantized_layers': {
                    experts.format(44): {'quant_algo': 'NVFP4', 'weight_scale_encoding': 'csf_32x2_v1'},
                    experts.format(45): {'quant_algo': 'W4A16_NVFP4', 'group_size': 16}}}}
        no_mtp = {'text_config': {'num_hidden_layers': 45},
                  'quantization_config': {'quantized_layers': {experts.format(44): {'quant_algo': 'NVFP4'}}}}
        nvfp4 = {'text_config': {'num_hidden_layers': 45},
                 'quantization_config': {'quant_method': 'compressed-tensors', 'config_groups': {}}}
        for config, receipt, ready in ((fp4l, None, True), (no_mtp, None, False), (nvfp4, None, False),
                                       ({}, None, False), (nvfp4, 864, True), (nvfp4, 863, False)):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(config=config, receipt=receipt):
                model = Path(tmp)
                (model/'config.json').write_text(json.dumps(config))
                if receipt is not None:
                    (model/'conversion.complete.json').write_text(json.dumps({'converted_matrices': receipt}))
                self.assertIs(serve.mtp_ready(model), ready)

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

    @staticmethod
    def rdma_sysfs(root, devices):
        """A /sys/class/infiniband tree: (name, PCI address, port state, IPv4 GID or None)."""
        pci = root/'pci'
        for dev, addr, state, ipv4 in devices:
            port = root/dev/'ports'/'1'
            (pci/addr).mkdir(parents=True)
            for sub in ('gids', 'gid_attrs/types'):
                (port/sub).mkdir(parents=True)
            (port/'state').write_text(state + '\n')
            (root/dev/'device').symlink_to(pci/addr)
            # Link-local GIDs at 0-1 (RoCE v1, v2), the IPv4-mapped ones at 2-3,
            # as a GB10 shows them; slot 4 exists but is unpopulated.
            gids = [('fe80:0000:0000:0000:4ebb:47ff:fee8:8334', 'IB/RoCE v1'),
                    ('fe80:0000:0000:0000:4ebb:47ff:fee8:8334', 'RoCE v2')]
            if ipv4:
                gids += [('0000:0000:0000:0000:0000:ffff:' + ipv4, 'IB/RoCE v1'),
                         ('0000:0000:0000:0000:0000:ffff:' + ipv4, 'RoCE v2')]
            for i, (gid, kind) in enumerate(gids):
                (port/'gids'/str(i)).write_text(gid + '\n')
                (port/'gid_attrs'/'types'/str(i)).write_text(kind + '\n')
            (port/'gid_attrs'/'types'/str(len(gids))).mkdir()  # reading it fails, like an empty slot

    def test_missing_hca_fails_and_explicit_rails_are_exact(self):
        missing = dict(self.environment)
        missing.pop('FABRIC_HCA', None)
        with self.assertRaises(ValueError):
            serve.launch(missing, self.profile)
        env = dict(self.environment, FABRIC_HCA='rocep1s0f1,roceP2p1s0f1')
        _, result = serve.launch(env, self.profile)
        self.assertEqual(result['ATLAS_RDMA_RAILS'], env['FABRIC_HCA'])
        self.assertEqual(result['NCCL_IB_HCA'], env['FABRIC_HCA'])

    def test_fabric_hca_lists_and_same_port_siblings(self):
        # GB10 attaches its ConnectX-7 through two PCIe domains, so one cable
        # appears as two RDMA devices (domain 0000 and 0002, same bus:dev.fn).
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self.rdma_sysfs(root, [('rocep1s0f0', '0000:01:00.0', '4: ACTIVE', '0a64:c002'),
                                   ('roceP2p1s0f0', '0002:01:00.0', '4: ACTIVE', '0a64:c102'),
                                   ('rocep1s0f1', '0000:01:00.1', '4: ACTIVE', '0a64:c401'),
                                   ('roceP2p1s0f1', '0002:01:00.1', '1: DOWN', '0a64:c501')])
            self.assertEqual(serve.fabric_hcas('rocep1s0f0', root), ['rocep1s0f0', 'roceP2p1s0f0'])
            self.assertEqual(serve.fabric_hcas('rocep1s0f1', root), ['rocep1s0f1'])
            self.assertEqual(serve.fabric_hcas('rocep1s0f0,roceP2p1s0f0', root),
                             ['rocep1s0f0', 'roceP2p1s0f0'])
            self.assertEqual(serve.fabric_hcas('rocep1s0f0', root, siblings=False), ['rocep1s0f0'])
            self.assertEqual(serve.fabric_hcas('mlx5_0', root/'missing'), ['mlx5_0'])
        argv, env = serve.launch(dict(self.environment, FABRIC_HCA='a0,b1'), self.profile)
        self.assertEqual(env['NCCL_IB_HCA'], 'a0,b1')
        self.assertEqual(env['ATLAS_RDMA_RAILS'], 'a0,b1')
        for bad in ('', 'a0,', 'a0,b1\nX=1', 'a0;b1', 'a0:1', 'a0,a0'):
            with self.subTest(hca=bad), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, FABRIC_HCA=bad), self.profile)

    def test_a_second_physical_port_is_not_a_sibling(self):
        # ASUS GX10 (Enntity/sparkglm#35): the other PCIe domain is a separate
        # physical port. Up but unaddressed, it is left out; addressed but not
        # cabled to the other Spark, FABRIC_SIBLINGS=0 leaves it out.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            self.rdma_sysfs(root, [('rocep1s0f0', '0000:01:00.0', '4: ACTIVE', 'c0a8:6401'),
                                   ('roceP2p1s0f0', '0002:01:00.0', '4: ACTIVE', None)])
            self.assertEqual(serve.fabric_hcas('rocep1s0f0', root), ['rocep1s0f0'])
        for value, rails in (('0', ['rocep1s0f0']), ('1', ['rocep1s0f0', 'sib'])):
            with self.subTest(siblings=value), mock.patch.object(
                    serve, 'fabric_hcas', lambda v, siblings: v.split(',') + (['sib'] if siblings else [])):
                _, env = serve.launch(dict(self.environment, FABRIC_HCA='rocep1s0f0',
                                           FABRIC_SIBLINGS=value), self.profile)
                self.assertEqual(env['ATLAS_RDMA_RAILS'], ','.join(rails))
                self.assertEqual(env['NCCL_IB_HCA'], ','.join(rails))
        for bad in ('', 'no', 'false', ' 0', '2'):
            with self.subTest(siblings=bad), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, FABRIC_SIBLINGS=bad), self.profile)

    def test_both_profiles_cache_prompt_prefixes(self):
        # Multi-turn agents resend the whole conversation every turn; without
        # prefix caching each turn re-prefills it from scratch.
        for name in ('4x512k', '4x1m', '8x128k'):
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
                for name in ('4x512k', '4x1m', '8x128k')]
        exact = ('ATLAS_GLM_SPARSE_PREFILL_PIPE', 'ATLAS_GLM_MOE_PREFILL_PERSIST',
                 'ATLAS_GLM_MOE_UNPERMUTE_VEC', 'ATLAS_GLM_INDEX_LOGITS_V2', 'ATLAS_GLM_INDEX_SPLIT',
                 'ATLAS_GLM_ROUTER_PREFILL_CUTLASS', 'ATLAS_GLM_MOE_DECODE_M16',
                 'ATLAS_GLM_MOE_DOWN_ZSKIP', 'ATLAS_GLM_PC_EVICT', 'ATLAS_GLM_PC_BRANCH',
                 'ATLAS_GLM_WARM_SKIP_CACHED', 'ATLAS_GLM_WARM_CHUNK_RUN', 'ATLAS_GLM_CMD_RDMA',
                 'ATLAS_GLM_MOE_DECODE_STREAM', 'ATLAS_GLM_DRAFT_TP', 'ATLAS_GLM_DECODE_FUSE',
                 'ATLAS_GLM_DECODE_GEMV_BATCH', 'ATLAS_GLM_MOE_DECODE_L2PF', 'ATLAS_GLM_MOE_STREAM_NOSYNC',
                 'ATLAS_GLM_DRAFT_TP_BATCH', 'ATLAS_RDMA_ONESHOT', 'ATLAS_RDMA_PAIR_CHAIN',
                 'ATLAS_GLM_DRAFT_TP_CTX', 'ATLAS_DFLASH_CTX_ASYNC_POS', 'ATLAS_GLM_STRICT_SPEC',
                 # 2026-10-17: same prompt logprobs and greedy text with each on.
                 'ATLAS_GLM_KDA_SIDE_PROJ_LT', 'ATLAS_GLM_KDA_CONV_PLANES', 'ATLAS_GLM_HC_PREFILL_FUSE2',
                 'ATLAS_GLM_SP_GATHER_OVERLAP')
        # Lossless but not bit-for-bit against the option off: the prefill queue
        # order, a verify width taken from the drafter's confidence (verify
        # numerics already depend on the width), and copy (prompt-lookup) drafts,
        # which change only how many tokens a step accepts (byte-identical at a
        # pinned width). The first draft of a request no longer reads a row the
        # previous request left behind.
        policy = {'ATLAS_PREFILL_SRPT': '1', 'ATLAS_DFLASH_CONF_WIDTH': '1',
                  'ATLAS_DFLASH_FIRST_APPEND': 'none', 'ATLAS_DFLASH_COPY_DRAFTS': '1',
                  'ATLAS_DFLASH_COPY_REPLY_MATCH': '16', 'ATLAS_DFLASH_COPY_MISS_MAX': '3'}
        held_back = ('ATLAS_GLM_MLA_KVB_MXFP8', 'ATLAS_GLM_INDEX_MXFP8', 'ATLAS_GLM_KV_SHARD',
                     'ATLAS_GLM_PC_FINISH_LEAF', 'ATLAS_GLM_VERIFY_GRAPH', 'ATLAS_GLM_STEP_FUSE',
                     'ATLAS_GLM_LAYER_FORK', 'ATLAS_GLM_L2_AHEAD',
                     'ATLAS_KV_NVME_DIR', 'ATLAS_GLM_KDA_PREFILL_LT_FP8_SPLITK1',
                     'ATLAS_GLM_TAIL_CUT_DEEP', 'ATLAS_GLM_ZERO_ROWS',
                     # Opt-in (docs/ENGINE.md): checkpoint-format serving is slower,
                     # packed prefill is not yet exact, the rest are untuned or diagnostics.
                     'ATLAS_GLM_FP4L_NATIVE', 'ATLAS_GLM_PACKED_PREFILL', 'ATLAS_DFLASH_WARM_CTX',
                     'ATLAS_TAIL_SPLIT_MIN_TOKENS', 'ATLAS_PREFILL_BURST', 'ATLAS_TTFT_TRACE')
        for env in envs:
            for key in exact:
                self.assertEqual(env.get(key), '1', key)
            # A mask: 3 = both bit-exact mHC seam twins (partial and finalize).
            self.assertEqual(env.get('ATLAS_GLM_HC_SEAM_ILP'), '3')
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

    def test_display_carveout_is_off_unless_switched_on(self):
        plain, _ = serve.launch(self.environment, self.profile)
        self.assertEqual(plain[0], '/usr/local/bin/spark')
        self.assertNotIn('display-carveout', plain)
        off, _ = serve.launch(dict(self.environment, SPARKGLM_DISPLAY_CARVEOUT='0'), self.profile)
        self.assertEqual(off, plain)
        on, _ = serve.launch(dict(self.environment, SPARKGLM_DISPLAY_CARVEOUT='1'), self.profile)
        self.assertEqual(on, ['/usr/local/bin/spark', 'display-carveout',
                              '--lock=/run/lock/sparkglm/display-carveout.lock', '--', *plain])
        for bad in ('yes', 'true', '2', ' 1'):
            with self.subTest(switch=bad), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, SPARKGLM_DISPLAY_CARVEOUT=bad), self.profile)

    def test_env_example_leaves_the_display_carveout_off(self):
        lines = (HERE.parent/'.env.example').read_text().splitlines()
        self.assertIn('#DISPLAY_CARVEOUT=1', lines)
        self.assertFalse([l for l in lines if l.startswith('DISPLAY_CARVEOUT')])

    def test_kv_shard_is_opt_in_in_the_image_and_runs_beside_the_disk_tier(self):
        _, env = serve.launch(self.environment, self.profile)
        self.assertNotIn('ATLAS_GLM_KV_SHARD', env)
        _, env = serve.launch(dict(self.environment, SPARKGLM_KV_SHARD='1'), self.profile)
        self.assertEqual((env['ATLAS_GLM_KV_SHARD'], env['ATLAS_GLM_KV_SHARD_COMPACT']), ('1', '1'))
        for bad in ('yes', '2', ' 1', ''):
            with self.subTest(switch=bad), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, SPARKGLM_KV_SHARD=bad), self.profile)
        _, env = serve.launch(dict(self.environment, SPARKGLM_KV_SHARD='1', SPARKGLM_PREFIX_CACHE_GB='48'),
                              self.profile)
        self.assertEqual((env['ATLAS_GLM_KV_SHARD'], env['ATLAS_KV_NVME_GB']), ('1', '24'))

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

    @staticmethod
    def fake_node(root):
        """Model root, RDMA sysfs and recording docker/stat/df for start-node.sh."""
        models, bin_dir = root/'models', root/'bin'
        for path in ('nvidia--GLM-5.3-Flash-FP4-Local/config.json', 'incoai--GLM-5.3-Flash-DFlash2/config.json'):
            (models/path).parent.mkdir(parents=True)
            (models/path).touch()
        bin_dir.mkdir()
        sysfs = root/'sysfs'
        (sysfs/'rocep1s0f0/device/net/enp1s0f0np0').mkdir(parents=True)
        (sysfs/'rocep1s0f0/ports/1').mkdir(parents=True)
        (sysfs/'rocep1s0f0/ports/1/state').write_text('4: ACTIVE')
        for dev in ('roceP2p1s0f0', 'rocep1s0f1', 'roceP2p1s0f1'):
            (sysfs/dev/'ports/1').mkdir(parents=True)
            (sysfs/dev/'ports/1/state').write_text('4: ACTIVE')
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
                env=dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', FABRIC_SYSFS=str(sysfs), **fake))
            log = (root/'docker.log').read_text() if (root/'docker.log').exists() else ''
            return run, [line for line in log.splitlines() if line.startswith('run ')]
        return models, start

    def test_start_node_serves_the_pinned_checkpoint_as_published(self):
        pin = re.search(r'^CHECKPOINT=([^@\s]+)@', (HERE.parent/'start.sh').read_text(), re.M).group(1)
        self.assertIn(f'checkpoint_name={pin.replace("/", "--")}\n', (HERE/'start-node.sh').read_text())
        with tempfile.TemporaryDirectory() as root:
            models, start = self.fake_node(Path(root))
            fp4l = (models/'nvidia--GLM-5.3-Flash-FP4-Local').resolve()
            run, docker_run = start('0')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn(f'--mount type=bind,src={fp4l},dst={fp4l},readonly ', docker_run[0])
            self.assertIn(f'-e MODEL_PATH={fp4l} ', docker_run[0])
            self.assertNotIn('atlas-overlay', docker_run[0])
            # The NVFP4 release is served through its verified MTP overlay.
            nvfp4 = models/'nvidia--GLM-5.3-Flash-NVFP4'
            nvfp4.mkdir()
            (nvfp4/'config.json').touch()
            run, docker_run = start('1', '--checkpoint', nvfp4.name)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('sparkglm-verified.json', run.stderr)
            self.assertEqual(docker_run, [])
            (models/'atlas-overlay').mkdir()
            (models/'atlas-overlay/sparkglm-verified.json').touch()
            run, docker_run = start('1', '--checkpoint', nvfp4.name)
            self.assertEqual(run.returncode, 0, run.stderr)
            overlay = (models/'atlas-overlay').resolve()
            self.assertIn(f'--mount type=bind,src={nvfp4.resolve()},dst={nvfp4.resolve()},readonly ', docker_run[0])
            self.assertIn(f'--mount type=bind,src={overlay},dst={overlay},readonly ', docker_run[0])
            self.assertIn(f'-e MODEL_PATH={overlay} ', docker_run[0])
            for bad in ('../models', 'a/b', '', '.hidden', 'missing'):
                run, docker_run = start('0', '--checkpoint', bad)
                with self.subTest(checkpoint=bad):
                    self.assertNotEqual(run.returncode, 0)
                    self.assertEqual(docker_run, [])

    def test_start_node_mounts_the_disk_prefix_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            models, start = self.fake_node(root)
            run, docker_run = start('0')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertNotIn('prefix-cache', docker_run[0].lower())
            self.assertIn('-e FABRIC_SIBLINGS=0 ', docker_run[0])
            run, docker_run = start('1', '--no-fabric-siblings')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('-e FABRIC_SIBLINGS=0 ', docker_run[0])
            self.assertIn('-e SPARKGLM_KV_SHARD=0 ', docker_run[0])
            run, docker_run = start('0', '--kv-shard')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('-e SPARKGLM_KV_SHARD=1 ', docker_run[0])
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
                    self.assertFalse((root/'docker.log').exists(), 'invalid cache must not stop a container')
                    self.assertIn('prefix-cache', run.stderr)
            for fake in ({'FSTYPE': 'tmpfs'}, {'FSTYPE': 'overlayfs'}, {'FREE_KB': str(47 * 1048576)}):
                run, docker_run = start('0', '--prefix-cache-dir', str(cache), **fake)
                with self.subTest(fake=fake):
                    self.assertNotEqual(run.returncode, 0)
                    self.assertEqual(docker_run, [])
                    self.assertFalse((root/'docker.log').exists(), 'invalid cache must not stop a container')
                    self.assertIn('prefix-cache', run.stderr)

    def test_profiles_are_selected_by_name(self):
        self.assertEqual(serve.profile_path({}, HERE), HERE/'profiles'/'4x512k.json')
        small = json.loads(serve.profile_path({'SPARKGLM_PROFILE': '8x128k'}, HERE).read_text())
        argv, _ = serve.launch(self.environment, small)
        self.assertIn('--max-seq-len=131072', argv)
        self.assertIn('--max-num-seqs=8', argv)
        # The model's full window, four requests sharing one pool; otherwise the 4x512k profile.
        full = json.loads(serve.profile_path({'SPARKGLM_PROFILE': '4x1m'}, HERE).read_text())
        base = json.loads(serve.profile_path({}, HERE).read_text())
        argv, _ = serve.launch(self.environment, full)
        self.assertIn('--max-seq-len=1048576', argv)
        self.assertIn('--max-num-seqs=4', argv)
        self.assertEqual(full['environment'], base['environment'])
        self.assertEqual([a for a in full['server_argv'] if not a.startswith('--max-seq-len=')],
                         [a for a in base['server_argv'] if not a.startswith('--max-seq-len=')])
        for bad in ('../x', '', 'a/b', '.hidden'):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                serve.profile_path({'SPARKGLM_PROFILE': bad}, HERE)

    def test_tp4_profile_is_4x1m_on_four_ranks(self):
        # Four Sparks in a daisy chain: 4x1m with the options the engine
        # refuses beyond two ranks switched off (its table in
        # serve_phases/preflight/glm_chain.rs), and NCCL left to startup traffic over sockets, since it cannot reach a
        # Spark that is not cabled to it.
        tp4 = json.loads(serve.profile_path({'SPARKGLM_PROFILE': '4x1m-tp4'}, HERE).read_text())
        full = json.loads(serve.profile_path({'SPARKGLM_PROFILE': '4x1m'}, HERE).read_text())
        expected = {k: '0' if k in REFUSED_BEYOND_TWO_RANKS else v for k, v in full['environment'].items()}
        # Checkpoint-format weight slices except KDA decode, which keeps
        # production's NVFP4 bytes under the TP2 shard's scale (an own-shard
        # re-quantization of a quarter matrix degrades DFlash verify at four
        # ranks); 8 MiB chain one-shots for the prefill seams; the chain's
        # proxies spinning through the vocab head's gap, and the piecewise
        # DFlash verify graphs, which capture the chain's all-reduces; the
        # drafter's MLP and head split across the ranks (its attention stays whole); and
        # CUTLASS blockwise FP8 for the native block-FP8 prefill GEMMs (cuBLASLt
        # has no 128-block FP8 kernel for SM121); and the MoE input gathered as
        # its routed NVFP4 rows (3.5x fewer bytes over the chain, same bits);
        # the MoE output's reduce-scatter overlapped with the next attention
        # gather, its unpermute on 32 blocks so the chain kernels keep SMs; and
        # the KDA conv writing FlashKDA's q/k/v planes (no pack copy, same bits);
        # and dead verify rows, so an owner batch skips the experts of drafts
        # not worth verifying (same greedy outputs); and the KDA o_proj on
        # this rank's rows after an MXFP8 all-to-all instead of a reduce-
        # scatter (one FP32 sum over all heads; prompt NLL unchanged); and the
        # MoE routed before the shared expert, so the router's weight prefetch
        # overlaps the mHC and norm kernels (same bits); and the KDA attention
        # input gathered as its MXFP8 cast plus side planes (same bits); and the
        # mHC decode finalize writing the next block's input norm (one launch
        # fewer per site, same words); and the MoE unpermute writing its rows
        # straight into one fused chain reduce-scatter in 16 MiB segments
        # (~80-96 MiB pinned per rank, same sums in the same order); and the
        # routed prefill MoE's down rows stored as E4M3 with per-64 scales
        # (lossy: prompt NLL and the 60-item needle gate indistinguishable).
        expected.update(NCCL_NET='Socket', NCCL_IB_DISABLE='1', ATLAS_RDMA_ONESHOT_STRIPE_MIN='16384',
                        ATLAS_GLM_FP4L_NATIVE='1', ATLAS_GLM_FP4L_KDA_MXFP8_DECODE='0',
                        ATLAS_RDMA_ONESHOT_MAX='8388544', ATLAS_RDMA_PAIR_SPIN_US='30000',
                        ATLAS_GLM_VERIFY_GRAPH='1', ATLAS_GLM_DRAFT_TP='mlp,head',
                        ATLAS_GLM_FP8B_CUTLASS='1', ATLAS_GLM_SP_MOE_QUANT_GATHER='1',
                        ATLAS_GLM_SP_MOE_SEAM='1', ATLAS_GLM_SP_MOE_SEAM_CTAS='32',
                        ATLAS_GLM_KDA_FLASH_PLANES='1', ATLAS_DFLASH_DEAD_ROWS='1',
                        ATLAS_GLM_SP_KDA_OPROJ_A2A='1', ATLAS_GLM_MOE_ROUTE_FIRST='1',
                        ATLAS_GLM_SP_KDA_QUANT_GATHER='1', ATLAS_GLM_HC_DECODE_NORM='1',
                        ATLAS_RDMA_RS_FUSED_SEG='16777216',
                        ATLAS_GLM_MOE_DOWN_E4M3='1')
        self.assertEqual(tp4['environment'], expected)
        four = {'--world-size=2': '--world-size=4', '--tp-size=2': '--tp-size=4', '--ep-size=2': '--ep-size=4'}
        self.assertEqual(tp4['server_argv'], [four.get(a, a) for a in full['server_argv']])
        for arg in four:
            self.assertIn(arg, full['server_argv'])

    def test_chain_ranks_take_rails_toward_each_neighbour(self):
        tp4 = json.loads(serve.profile_path({'SPARKGLM_PROFILE': '4x1m-tp4'}, HERE).read_text())
        left, right = 'rocep1s0f0,roceP2p1s0f0', 'rocep1s0f1,roceP2p1s0f1'
        chain = dict(self.environment, FABRIC_HCA='', FABRIC_INTERFACE='enP2p1s0f0np0')
        sides = {'0': {'FABRIC_RIGHT_HCA': right}, '1': {'FABRIC_LEFT_HCA': left, 'FABRIC_RIGHT_HCA': right},
                 '2': {'FABRIC_LEFT_HCA': left, 'FABRIC_RIGHT_HCA': right}, '3': {'FABRIC_LEFT_HCA': left}}
        for rank, rails in sides.items():
            with self.subTest(rank=rank):
                argv, env = serve.launch(dict(chain, NODE_RANK=rank, **rails), tp4)
                self.assertIn('--world-size=4', argv)
                self.assertIn(f'--rank={rank}', argv)
                self.assertEqual(env.get('ATLAS_RDMA_LEFT_RAILS'), rails.get('FABRIC_LEFT_HCA'))
                self.assertEqual(env.get('ATLAS_RDMA_RIGHT_RAILS'), rails.get('FABRIC_RIGHT_HCA'))
                self.assertNotIn('ATLAS_RDMA_RAILS', env)
                self.assertNotIn('NCCL_IB_HCA', env)
                self.assertEqual(env['NCCL_SOCKET_IFNAME'], 'enP2p1s0f0np0')
                self.assertEqual(env['NCCL_NET'], 'Socket')
        bad = [('0', {'FABRIC_LEFT_HCA': left, 'FABRIC_RIGHT_HCA': right}),  # nothing left of rank 0
               ('3', {'FABRIC_LEFT_HCA': left, 'FABRIC_RIGHT_HCA': right}),  # nothing right of rank 3
               ('1', {'FABRIC_RIGHT_HCA': right}), ('2', {'FABRIC_LEFT_HCA': left}),
               ('4', {'FABRIC_LEFT_HCA': left}), ('1', {'FABRIC_LEFT_HCA': 'a0,a0', 'FABRIC_RIGHT_HCA': right}),
               ('1', {'FABRIC_LEFT_HCA': left, 'FABRIC_RIGHT_HCA': right, 'FABRIC_HCA': left})]
        for rank, rails in bad:
            with self.subTest(rank=rank, rails=rails), self.assertRaises(ValueError):
                serve.launch(dict(chain, NODE_RANK=rank, **rails), tp4)
        # A two-rank profile takes no chain rails.
        with self.assertRaises(ValueError):
            serve.launch(dict(self.environment, FABRIC_RIGHT_HCA=right), self.profile)

    def test_start_node_launches_a_chain_rank(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            _, start = self.fake_node(root)
            left, right = 'rocep1s0f0,roceP2p1s0f0', 'rocep1s0f1,roceP2p1s0f1'
            run, docker_run = start('2', '--world-size', '4', '--left-hca', left, '--right-hca', right)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('-e NODE_RANK=2 ', docker_run[0])
            self.assertIn(f'-e FABRIC_LEFT_HCA={left} ', docker_run[0])
            self.assertIn(f'-e FABRIC_RIGHT_HCA={right} ', docker_run[0])
            self.assertIn('-e FABRIC_HCA= ', docker_run[0])
            run, docker_run = start('3', '--world-size', '4', '--left-hca', right)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('-e FABRIC_RIGHT_HCA= ', docker_run[0])
            for rank, extra in (('4', ['--world-size', '4', '--left-hca', left]),
                                ('1', ['--world-size', '3', '--left-hca', left, '--right-hca', right]),
                                ('1', ['--world-size', '4', '--right-hca', right]),
                                ('1', ['--world-size', '4', '--left-hca', 'rocep1s0f9', '--right-hca', right]),
                                ('1', ['--world-size', '4', '--fabric-hca', left, '--left-hca', left,
                                       '--right-hca', right])):
                run, docker_run = start(rank, *extra)
                with self.subTest(rank=rank, extra=extra):
                    self.assertNotEqual(run.returncode, 0)
                    self.assertEqual(docker_run, [])
            # Two ranks take no chain rails.
            run, docker_run = start('1', '--right-hca', right)
            self.assertNotEqual(run.returncode, 0)

    def test_invalid_cluster_configuration_fails_before_process_launch(self):
        for key, value in [('NODE_RANK', '2'), ('MASTER_ADDR', 'not-an-address'),
                           ('MASTER_PORT', '0'), ('FABRIC_INTERFACE', 'eth0\nINJECTED=1'),
                           ('MODEL_PATH', 'relative'), ('DRAFTER_PATH', 'rel'),
                           ('SERVED_MODEL_NAME', 'bad\nname')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                serve.launch(dict(self.environment, **{key:value}), self.profile)


if __name__ == '__main__':
    unittest.main()
