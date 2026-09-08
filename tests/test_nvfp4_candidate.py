#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline artifact and actual rank-entrypoint regression tests."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research/experiments/nvfp4-tp2"
spec = importlib.util.spec_from_file_location("nvfp4_prepare", EXP / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
LOCK = json.loads((EXP / "candidate.json").read_text())


class CandidateTest(unittest.TestCase):
    def test_experiment_sources_compile(self):
        for path in EXP.rglob("*.py"):
            compile(path.read_text(), str(path), "exec")
        for path in EXP.glob("*.sh"):
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_matrix_reuses_existing_harness_and_calibrates_tokens(self):
        result = subprocess.run(["python3", str(EXP / "matrix.py"), "--base-url", "http://127.0.0.1:8889",
                                 "--model", "test", "--arm", "nvfp4", "--output", "/tmp/results"],
                                text=True, capture_output=True, check=True)
        lines = [line for line in result.stdout.splitlines() if not line.startswith("#")]
        self.assertEqual(len(lines), 28)
        self.assertTrue(all("benchmarks/staggered_openai.py" in line and "--exact-prompt-tokens" in line for line in lines))
        self.assertEqual(sum("--prompt-token-list 512,32768" in line for line in lines), 4)

    def args(self, rank=0):
        return argparse.Namespace(rank=rank, image_id="sha256:" + "a" * 64,
                                  head_ip="192.0.2.1", worker_ip="192.0.2.2",
                                  interface="eth1", hca="mlx5_0", gid_index=3,
                                  models=Path("/srv/models"), cache=Path("/srv/cache/nvfp4"))

    def test_copied_files_and_pins(self):
        for row in LOCK["files"]:
            self.assertEqual(hashlib.sha256((EXP / row["path"]).read_bytes()).hexdigest(), row["sha256"])
        self.assertIn(LOCK["base_image"], (EXP / "Dockerfile").read_text())
        for model in LOCK["models"].values():
            self.assertRegex(model["revision"], r"^[a-f0-9]{40}$")

    def test_plan_never_executes_and_has_no_lifecycle_side_effects(self):
        for rank in (0, 1):
            result = prepare.plan(self.args(rank))
            command = result["docker_argv"]
            self.assertNotIn("ssh", command)
            self.assertNotIn("rm", command)
            self.assertIn("--restart", command)
            self.assertEqual(command[command.index("--restart") + 1], "no")
            self.assertIn("--revision", result["download_commands"][0])
            self.assertEqual(result["environment"]["MOE_BACKEND"], "humming")
            self.assertEqual(result["environment"]["KV_CACHE_DTYPE"], "fp8")
        bad = self.args()
        bad.image_id = "sparkglm:latest"
        with self.assertRaises(ValueError):
            prepare.plan(bad)

    def test_checkpoint_rejects_missing_shards_and_config_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ("a" * 40)
            path.mkdir()
            config = {"model_type": "glm5_next", "quantization_config": {"quant_method": "compressed-tensors"}}
            model = {"revision": path.name, "config_sha256": prepare.config_digest(config)}
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                prepare.validate_weights(path, model)
            (path / "model.safetensors").write_bytes(b"fixture, not weights")
            prepare.validate_weights(path, model)
            (path / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                prepare.validate_weights(path, model)

    def test_actual_shell_entrypoints_produce_matching_serve_contract(self):
        # Execute the real copied shell bodies with a recording vllm executable.
        # No GPU, Docker, server, network or upstream patch code is executed.
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            (tmp / "config.json").write_text("{}")
            stub = tmp / "vllm"
            stub.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            stub.chmod(0o755)
            commands = []
            for rank, role in enumerate(("head", "worker")):
                env = dict(os.environ, **prepare.plan(self.args(rank))["environment"])
                env.update(PATH=str(tmp) + os.pathsep + os.environ["PATH"], MODEL_DIR=str(tmp))
                result = subprocess.run(["bash", str(EXP / (role + ".sh"))], env=env, text=True, capture_output=True, check=True)
                command = json.loads(result.stdout.splitlines()[-1])
                commands.append(command)
                self.assertEqual(command[:2], ["serve", str(tmp)])
                for flag, value in (("--tensor-parallel-size", "2"), ("--nnodes", "2"),
                                    ("--moe-backend", "humming"), ("--kv-cache-dtype", "fp8"),
                                    ("--max-num-batched-tokens", "1024"), ("--node-rank", str(rank))):
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertNotIn("--quantization", command)  # checkpoint auto-detection
                self.assertNotIn("--enforce-eager", command)
                drafter = json.loads(command[command.index("--speculative-config") + 1])
                self.assertEqual(drafter["num_speculative_tokens"], 7)
                self.assertEqual(drafter["draft_tensor_parallel_size"], 1)
                self.assertEqual(drafter["model"], "/models/draft")
            self.assertIn("--headless", commands[1])
            self.assertEqual(commands[0][commands[0].index("--host") + 1], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
