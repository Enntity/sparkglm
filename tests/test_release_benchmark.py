# SPDX-License-Identifier: AGPL-3.0-only
"""Require the public SparkModelBench scorecard for the current product version."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseBenchmark(unittest.TestCase):
    def test_current_release_has_a_versioned_four_lane_scorecard(self):
        version = json.loads((ROOT / "install/release.json").read_text())["version"]
        bundle = ROOT / "results" / f"sparkmodelbench-v{version}"
        path = bundle / "CARD.md"
        self.assertTrue(path.is_file(), f"Run bench/run.sh and publish {path.relative_to(ROOT)}")
        card = path.read_text()
        self.assertIn(f"# SparkGLM v{version} · SparkModelBench\n", card.split("## ")[0])
        self.assertRegex((bundle / "SHA256SUMS").read_text(), r"(?m)^[0-9a-f]{64}  CARD\.md$")
        sections = {}
        for section in card.split("\n## ")[1:]:
            heading, _, body = section.partition("\n")
            sections[heading] = body.splitlines()
        for heading in ("RigMark Quality · task completion", "RigMark · generation, median",
                        "SparkGLM · staggered C4", "RigMark · prefill"):
            with self.subTest(heading=heading):
                rows = [line for line in sections.get(heading, []) if line.startswith("|")]
                self.assertGreater(len(rows), 2, f"Missing measured rows for {heading}")
