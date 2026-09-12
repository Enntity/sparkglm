#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Portable profile checks; optional exact two-rank comparison to private evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from render import HERE, render, verify_local_artifacts

VALUES = dict(rank="0", master="192.0.2.1", bind="127.0.0.1", port="8893",
              nic="fabric0", hca="roce0", name="atlas-r23-r0", image="atlas-r23:local",
              overlay="/srv/models/overlay", original="/srv/models/original",
              cuda_cache="/srv/cuda-cache", template="/srv/chat_template.jinja",
              flash_library_dir="/srv/atlas-flash", native_library="/srv/atlas-sparse/native.so")

def runtime(argv):
    start = argv.index("-i") + 1
    end = start
    while "=" in argv[end] and not argv[end].startswith("/"):
        end += 1
    env = dict(item.split("=", 1) for item in argv[start:end])
    if len(env) != end-start:
        raise ValueError("duplicate environment key")
    return env, argv[argv.index("/usr/local/bin/spark") + 1:]

class ProfileTests(unittest.TestCase):
    def test_both_ranks_keep_r23_flags_and_guards(self):
        for rank in ("0", "1"):
            argv = render(dict(VALUES, rank=rank)); env, server = runtime(argv)
            self.assertEqual(shlex.split(shlex.join(argv)), argv)
            for flag in ("--memory=114g", "--memory-swap=114g", "--restart=no", "--ipc=private"):
                self.assertIn(flag, argv)
            for flag in (f"--rank={rank}", "--gpu-memory-utilization=0.914", "--max-seq-len=32768",
                         "--max-prefill-tokens=4096", "--max-num-seqs=4", "--max-batch-size=4",
                         "--num-drafts=2", "--oom-guard-mb=4096", "--mtp-gate=force"):
                self.assertIn(flag, server)
            for key in ("ATLAS_GLM_SPARSE_DECODE_SPLIT", "ATLAS_GLM_SPARSE_DECODE_TC",
                        "ATLAS_GLM_SPARSE_NATIVE", "ATLAS_KDA_FLASH_PREFILL", "ATLAS_GLM_HC_TF32_PREWARM",
                        "ATLAS_GLM_K3_HEAD_BATCHM", "ATLAS_GLM_K3_MLA_O_BATCHM", "ATLAS_GLM_C3_GROUPED_MOE",
                        "ATLAS_GLM_MTP_REPAIR", "ATLAS_GLM_MTP_LONG_CONTEXT"):
                self.assertEqual(env[key], "1")
            self.assertEqual(env["ATLAS_KV_OVERCOMMIT"], "0")
            self.assertNotIn("ATLAS_GLM_K3_MLA_O_COMPARE", env)
            self.assertNotIn("ATLAS_GLM_ROUTE_CAPTURE", env)
            self.assertFalse(any("${" in value or "nsys" in value for value in argv))

    def test_artifact_requirements_and_distinct_source_identities(self):
        p = json.loads((HERE / "profile.json").read_text())
        self.assertEqual(p["source_revision"], "9b3e316a7f5ca92ec22a68414e7aefeda32023dd")
        self.assertEqual(p["latest_diagnostic_source_revision"], "faf4e874b2d418b67a5bc4ec5793c450c9f27ebb")
        self.assertFalse(p["latest_diagnostic_is_measured_profile"])
        self.assertEqual(len(p["external_artifacts"]), 3)
        self.assertEqual(len({x["sha256"] for x in p["external_artifacts"]}), 3)
        self.assertTrue(any(x["filename"] == "libatlas_glm53_flash_kda.so" for x in p["external_artifacts"]))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                verify_local_artifacts(dict(VALUES, template=str(Path(d) / "missing")))
            bad = Path(d) / "template"; bad.write_text("not the retained template")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_local_artifacts(dict(VALUES, template=str(bad)))

    def test_invalid_missing_and_unknown_parameters_fail(self):
        for changed in ({"rank": "2"}, {"original": "relative"}, {"native_library": "/tmp/a,readonly"},
                        {"flash_library_dir": ""}, {"image": "x\nfoo"}, {"port": "65536"}, {"extra": "x"}):
            with self.assertRaises(ValueError): render(dict(VALUES, **changed))
        with self.assertRaises(ValueError): render({k:v for k,v in VALUES.items() if k != "native_library"})


def compare_receipt(path):
    raw = path.read_bytes(); profile = json.loads((HERE / "profile.json").read_text())
    if hashlib.sha256(raw).hexdigest() != profile["verified_manifest_sha256"]:
        raise ValueError("original manifest hash mismatch")
    commands = json.loads(raw)["commands"]
    if len(commands) != 2: raise ValueError("exactly two original ranks required")
    ranks = []
    for command in commands:
        old = command["argv"]; env, server = runtime(old)
        rank = next(x.split("=",1)[1] for x in server if x.startswith("--rank=")); ranks.append(rank)
        mounts = [old[i+1] for i,x in enumerate(old) if x == "--mount"]
        source = lambda value: dict(x.split("=",1) for x in value.split(",") if "=" in x)["src"]
        val = dict(VALUES, rank=rank, master=next(x.split("=",1)[1] for x in server if x.startswith("--master-addr=")),
                   bind=next(x.split("=",1)[1] for x in server if x.startswith("--bind=")),
                   port=next(x.split("=",1)[1] for x in server if x.startswith("--port=")),
                   name=command["container"], image=old[old.index("-i")-1], hca=env["NCCL_IB_HCA"],
                   nic=env["NCCL_SOCKET_IFNAME"], overlay=source(mounts[0]), original=source(mounts[1]),
                   cuda_cache=source(mounts[2]), template=source(mounts[3]))
        new = render(val)
        # Portable libraries replace image-embedded external artifacts; no compute flags change.
        expected_artifacts = [x.replace("${flash_library_dir}", val["flash_library_dir"]).replace("${native_library}", val["native_library"])
                              for x in profile["artifact_mounts"]]
        def strip(argv, artifact_mounts=False):
            result=[]; i=0
            while i<len(argv):
                if argv[i]=="--mount" and ((artifact_mounts and argv[i+1] in expected_artifacts) or
                        (not artifact_mounts and any(s in argv[i+1] for s in ("dst=/profile", "dst=/opt/nvidia/nsight-systems")))):
                    i+=2; continue
                if argv[i].endswith("/nsys"):
                    i=argv.index("/usr/local/bin/spark",i); continue
                result.append(argv[i]); i+=1
            return result
        if strip(old) != strip(new, True): raise ValueError(f"rank{rank}: exact argv drift")
    if set(ranks)!={"0","1"}: raise ValueError("missing rank")
    print("PASS exact r23 Docker guards, environment and server argv on both ranks; only profiling removed and operator artifact mounts added")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--verified-manifest", type=Path); args=parser.parse_args()
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    if args.verified_manifest: compare_receipt(args.verified_manifest)
