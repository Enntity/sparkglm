# SPDX-License-Identifier: AGPL-3.0-only
"""The public mode selects a complete precision policy before engine launch."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mode_serve", HERE / "serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class ServingModes(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((HERE / "profiles/4x1m.json").read_text())
        self.env = {
            "NODE_RANK": "0", "MASTER_ADDR": "192.0.2.1",
            "FABRIC_INTERFACE": "fabric0", "FABRIC_HCA": "roce0",
            "MODEL_PATH": "/models/checkpoint", "DRAFTER_PATH": "/models/drafter",
            "SPARKGLM_PREFIX_CACHE_GB": "48",
        }

    def launch(self, mode=None):
        env = dict(self.env)
        if mode is not None:
            env["SPARKGLM_MODE"] = mode
        return serve.launch(env, self.profile)

    def test_default_strict_preserves_native_decode_and_reference_activation_policy(self):
        _, env = self.launch()
        self.assertEqual(env["ATLAS_GLM_SERVING_MODE"], "strict")
        self.assertEqual(env["ATLAS_GLM_FP4L_NATIVE"], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_KDA_MXFP8_DECODE"], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_DECODE_A16"], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_INPUT_SCALES"], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_SHARED_FP8"], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_DENSE_MXFP8"], "1")
        self.assertEqual(env["ATLAS_NO_DFLASH_DRAFTER_NVFP4"], "1")
        for flag in ("ATLAS_DFLASH_NVFP4_TC", "ATLAS_DFLASH_NVFP4_HEAD",
                     "ATLAS_DFLASH_MXFP8", "ATLAS_DFLASH_MXFP8_HEAD",
                     "ATLAS_DFLASH_DROP_BF16", "ATLAS_DFLASH_CTX_NVFP4"):
            self.assertEqual(env[flag], "0", flag)

    def test_spark_uses_qualified_target_conversions(self):
        _, env = self.launch("spark")
        self.assertEqual(env["ATLAS_GLM_SERVING_MODE"], "spark")
        self.assertEqual(env["ATLAS_GLM_FP4L_KDA_MXFP8_DECODE"], "0")
        for group in ("MLA_FP8", "KDA_W8A8_PREFILL"):
            self.assertEqual(env["ATLAS_GLM_FP4L_" + group], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_SHARED_FP8"], "0")
        self.assertEqual(env["ATLAS_GLM_FP4L_DENSE_MXFP8"], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_DECODE_A16"], "0")
        self.assertEqual(env["ATLAS_DFLASH_NVFP4_TC"], "1")
        self.assertEqual(env["ATLAS_GLM_DECODE_FUSE"], "1")
        self.assertNotIn("ATLAS_NO_DFLASH_DRAFTER_NVFP4", env)

    def test_spark_disables_shared_split_rounding_that_failed_arithmetic(self):
        for profile in ("4x1m", "4x512k", "8x128k"):
            self.profile = json.loads((HERE / f"profiles/{profile}.json").read_text())
            _, env = self.launch("spark")
            self.assertEqual(env.get("ATLAS_GLM_SHARED_TP_SPLIT"), "0")
            self.assertEqual(env["ATLAS_GLM_FP4L_KDA_MXFP8_DECODE"], "0")
            self.assertEqual(env["ATLAS_GLM_FP4L_SHARED_FP8"], "0")

    def test_shared_and_dense_overrides_are_spark_only(self):
        for flag in ('ATLAS_GLM_FP4L_SHARED_FP8',):
            for value in ('0', '1'):
                self.profile['environment'][flag] = value
                _, env = self.launch('spark')
                self.assertEqual(env[flag], value)
                if value == '0':
                    with self.assertRaises(ValueError):
                        self.launch('strict')
            self.profile['environment'][flag] = 'invalid'
            with self.assertRaises(ValueError):
                self.launch('spark')
            del self.profile['environment'][flag]

    def test_unqualified_group_stays_native_in_spark(self):
        for flag in ('ATLAS_GLM_FP4L_DENSE_MXFP8',):
            self.profile['environment'][flag] = '0'
            with self.assertRaises(ValueError):
                self.launch('spark')
            del self.profile['environment'][flag]

    def test_modes_are_explicit_and_do_not_mutate_the_capacity_profile(self):
        before = json.dumps(self.profile, sort_keys=True)
        for bad in ("", "fast", "STRICT", "../spark"):
            with self.subTest(mode=bad), self.assertRaises(ValueError):
                self.launch(bad)
        self.launch("strict")
        self.launch("spark")
        self.assertEqual(json.dumps(self.profile, sort_keys=True), before)

    def test_disk_cache_cannot_cross_modes(self):
        _, strict = self.launch("strict")
        _, spark = self.launch("spark")
        for flag in ("ATLAS_KV_NVME_DIR", "ATLAS_SSM_TIER_SWAP_DIR"):
            self.assertNotEqual(strict[flag], spark[flag], flag)
            self.assertIn("/strict/", strict[flag])
            self.assertIn("/spark/", spark[flag])

    def test_unreviewed_target_precision_overrides_are_rejected(self):
        self.profile["environment"]["ATLAS_GLM_FP4L_MLA_FP8"] = "0"
        for mode in ("strict", "spark"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "MLA_FP8"):
                self.launch(mode)

    def test_strict_rejects_alternative_speculative_decoders(self):
        for flag in ('--ngram-speculative', '--self-speculative=true'):
            with self.subTest(flag=flag):
                self.profile['server_argv'].append(flag)
                with self.assertRaisesRegex(ValueError, 'serial'):
                    self.launch('strict')
                self.profile['server_argv'].pop()

    def test_native_mtp_keeps_normal_capacity_cache_and_target_precision(self):
        self.env["SPARKGLM_STRICT_MTP"] = "1"
        args, env = self.launch("strict")
        self.assertIn("--speculative", args)
        self.assertNotIn("--dflash", args)
        self.assertFalse(any(a.startswith("--draft-model") for a in args))
        self.assertIn("--max-batch-size=4", args)
        self.assertIn("--max-num-seqs=4", args)
        self.assertIn("--max-seq-len=1048576", args)
        self.assertIn("--enable-prefix-caching", args)
        self.assertIn("--max-prefill-tokens=8192", args)
        self.assertEqual(env["ATLAS_GLM_MTP_LONG_CONTEXT"], "1")
        self.assertEqual(env["ATLAS_GLM_STRICT_NATIVE_MTP"], "1")
        self.assertEqual(env["ATLAS_GLM_LONG_BATCH_VERIFY"], "0")
        self.assertEqual(env["ATLAS_GLM_SPARSE_NATIVE"], self.profile["environment"]["ATLAS_GLM_SPARSE_NATIVE"])
        self.assertEqual(env["ATLAS_GLM_K3_MLA_QUERY_BATCHM"], self.profile["environment"]["ATLAS_GLM_K3_MLA_QUERY_BATCHM"])
        self.assertEqual(env["ATLAS_GLM_K3_MLA_O_BATCHM"], self.profile["environment"]["ATLAS_GLM_K3_MLA_O_BATCHM"])
        self.assertEqual(env["ATLAS_GLM_SPARSE_DECODE_TC"], self.profile["environment"]["ATLAS_GLM_SPARSE_DECODE_TC"])
        self.assertEqual(env["ATLAS_GLM_SPARSE_DECODE_SPLIT"], self.profile["environment"]["ATLAS_GLM_SPARSE_DECODE_SPLIT"])
        self.assertEqual(env["ATLAS_GLM_FP4L_KDA_MXFP8_DECODE"], "1")
        self.assertEqual(env["ATLAS_GLM_FP4L_DECODE_A16"], "1")
        with self.assertRaises(ValueError):
            self.launch("spark")
        self.env["SPARKGLM_STRICT_MTP"] = "yes"
        with self.assertRaises(ValueError):
            self.launch("strict")

    def test_native_mtp_one_million_profile_preserves_host_headroom(self):
        self.env["SPARKGLM_STRICT_MTP"] = "1"
        args, env = self.launch("strict")
        self.assertEqual(env.get("ATLAS_KV_MAX_BLOCKS"), "66000")
        self.assertIn("--max-seq-len=1048576", args)
        self.assertIn("--oom-guard-mb=4096", args)
        for profile in ("8x128k", "4x512k"):
            self.profile = json.loads((HERE / f"profiles/{profile}.json").read_text())
            _, env = self.launch("strict")
            self.assertNotIn("ATLAS_KV_MAX_BLOCKS", env)
        self.profile = json.loads((HERE / "profiles/4x1m.json").read_text())
        self.env["SPARKGLM_STRICT_MTP"] = "0"
        for mode in ("strict", "spark"):
            _, env = self.launch(mode)
            self.assertNotIn("ATLAS_KV_MAX_BLOCKS", env)

    def test_explicit_modes_are_tp2_only(self):
        self.profile["server_argv"] = [
            "--world-size=4" if arg == "--world-size=2" else arg
            for arg in self.profile["server_argv"]
        ]
        self.env.update(FABRIC_HCA="", FABRIC_RIGHT_HCA="roce1")
        with self.assertRaisesRegex(ValueError, "TP2"):
            self.launch("spark")


if __name__ == "__main__":
    unittest.main()
