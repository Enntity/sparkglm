#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only profile tests; optionally compare the original private receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

from render import render

HERE = Path(__file__).resolve().parent
VALUES = dict(rank="0", master="192.0.2.1", bind="127.0.0.1", port="8893",
              nic="fabric0", hca="roce0", name="atlas-test-r0", image="atlas-test:local",
              overlay="/srv/models/overlay", original="/srv/models/original",
              cuda_cache="/srv/cuda-cache", template="/srv/chat_template.jinja")


def runtime(argv):
    start = argv.index("-i") + 1
    end = start
    while "=" in argv[end] and not argv[end].startswith("/"):
        end += 1
    environment = dict(item.split("=", 1) for item in argv[start:end])
    server = argv[argv.index("/usr/local/bin/spark") + 1:]
    return environment, server


class ProfileTests(unittest.TestCase):
    def test_render_retains_guards_and_distinct_ranks(self):
        for rank in ("0", "1"):
            argv = render(dict(VALUES, rank=rank))
            env, server = runtime(argv)
            self.assertEqual(argv[:3], ["docker", "run", "-d"])
            for flag in ("--memory=114g", "--memory-swap=114g", "--restart=no"):
                self.assertIn(flag, argv)
            for flag in (f"--rank={rank}", "--gpu-memory-utilization=0.914",
                         "--max-seq-len=32768", "--max-num-seqs=4",
                         "--max-batch-size=4", "--num-drafts=2", "--oom-guard-mb=4096"):
                self.assertIn(flag, server)
            self.assertNotIn("ATLAS_FP4_PREFILL", env)
            self.assertEqual(env["ATLAS_KV_OVERCOMMIT"], "0")
            self.assertEqual(env["ATLAS_GLM_MTP_REPAIR"], "1")
            self.assertEqual(env["ATLAS_GLM_MTP_LONG_CONTEXT"], "1")
            self.assertFalse(any("${" in value for value in argv))

    def test_unknown_or_unsafe_arguments_fail_closed(self):
        for changed in ({"rank": "2"}, {"original": "relative"},
                        {"overlay": "/tmp/a,readonly"}, {"image": ""}):
            with self.assertRaises(ValueError):
                render(dict(VALUES, **changed))


def compare_receipt(path):
    raw = path.read_bytes()
    profile = json.loads((HERE / "profile.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == profile["verified_manifest_sha256"]
    for command in json.loads(raw)["commands"]:
        old_env, old_server = runtime(command["argv"])
        rank = next(x.split("=", 1)[1] for x in old_server if x.startswith("--rank="))
        new_env, new_server = runtime(render(dict(VALUES, rank=rank)))
        # Only node-local transport selectors and network locators vary.
        for key in ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME"):
            old_env.pop(key)
            new_env.pop(key)
        assert new_env == old_env, "runtime environment drift"
        def normalized(args):
            return [x for x in args if not x.startswith(("--master-addr=", "--bind=", "--port="))]
        assert normalized(new_server) == normalized(old_server), "server argv drift"
        # Fixed Docker isolation/memory controls are not excluded with mounts.
        for option in profile["docker_options"]:
            if option not in ("--name", "${name}"):
                assert option in command["argv"], f"Docker guard drift: {option}"
    print("PASS exact verified r11b env/server flags; only site locators and profiling wrapper excluded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-manifest", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.verified_manifest:
        compare_receipt(args.verified_manifest)
