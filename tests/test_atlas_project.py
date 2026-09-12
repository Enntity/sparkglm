#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Invoke the separately licensed Atlas public-safe checks as subprocesses."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "research" / "atlas" / "project"
EXPERIMENTS = ROOT / "research" / "atlas" / "experiments" / "day2"


class AtlasProjectChecks(unittest.TestCase):
    def run_python(self, args, *, environment=None, minimum_tests=None):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        # Only the explicit public helper candidate may be loaded by this check.
        env.pop("PROFILE_EVENT_CANDIDATE", None)
        if environment:
            env.update(environment)
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        if minimum_tests is not None:
            match = re.search(r"Ran (\d+) tests?\b", output)
            self.assertIsNotNone(match, "Missing unittest execution count:\n" + output)
            self.assertGreaterEqual(int(match.group(1)), minimum_tests, output)
            self.assertIn("OK", output)
        return result.stdout

    def test_portable_profile(self):
        self.run_python([PROJECT / "test_profile.py"], minimum_tests=3)

    def test_source_reconstruction_guards(self):
        # Optional real-upstream reconstruction is deliberately separate from G0.
        self.run_python([PROJECT / "test_reconstruct.py", "-v"], minimum_tests=8)

    def test_standalone_source_hashes_syntax_and_include_closure(self):
        output = self.run_python([EXPERIMENTS / "verify-sources.py"])
        receipt = json.loads(output)
        self.assertIs(receipt["passed"], True)
        self.assertIs(receipt["gpu_run"], False)
        self.assertGreater(receipt["files"], 0)
        self.assertGreater(receipt["quoted_includes"], 0)

    def test_pure_route_count_helper(self):
        self.run_python(["-m", "unittest", "discover", "-s",
                         EXPERIMENTS / "route-analysis-helper", "-v"], minimum_tests=9)

    def test_pure_profile_interval_helper(self):
        directory = EXPERIMENTS / "profile-event-helper"
        self.run_python(["-m", "unittest", "discover", "-s", directory, "-v"],
                        environment={"PROFILE_EVENT_CANDIDATE": str(directory / "ds-reviewed.py")},
                        minimum_tests=10)

    def test_synthetic_joint_fixture_metadata(self):
        output = self.run_python([EXPERIMENTS / "moe-joint-m12" / "fixture-plan.py"])
        receipt = json.loads(output)
        self.assertEqual(receipt["status"], "LOCAL_PLAN_METADATA_CHECKS_PASS")
        self.assertIs(receipt["cuda_build"], False)
        self.assertIs(receipt["gpu_run"], False)


if __name__ == "__main__":
    unittest.main()
