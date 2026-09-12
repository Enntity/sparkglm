# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only reconstruction guards; synthetic Git data, no network/GPU."""
import argparse
import os
import sys
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest

SOURCE_REPO = None
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("atlas_reconstruct", HERE / "reconstruct.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.repo = self.root / "upstream"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "Synthetic test")
        git(self.repo, "config", "user.email", "test@example.invalid")
        (self.repo / "source.txt").write_text("base\n")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "source.txt").write_text("candidate\n")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "candidate")
        self.tip = git(self.repo, "rev-parse", "HEAD")
        patch = subprocess.check_output(["git", "-C", str(self.repo), "diff", "--full-index", self.base, self.tip])
        (self.root / "change.patch").write_bytes(patch)
        self.data = {
            "schema": "sparkglm.atlas-source-export/v2",
            "upstream": {"revision": self.base},
            "engine_revision": self.tip,
            "reconstructed_tree": git(self.repo, "rev-parse", "HEAD^{tree}"),
            "patch_chain": [{"path": "change.patch", "sha256": hashlib.sha256(patch).hexdigest()}],
            "changed_file_count": 1,
            "changed_files_sha256": {"source.txt": hashlib.sha256(b"candidate\n").hexdigest()},
        }
        self.manifest = self.root / "source.json"
        self.dest = self.root / "destination"
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.data))

    def run_prepare(self):
        return r.prepare(self.manifest, self.repo, self.dest)

    def test_exact_reconstruction_keeps_source_clean(self):
        before = git(self.repo, "rev-parse", "HEAD")
        receipt = self.run_prepare()
        self.assertEqual(receipt["tree"], self.data["reconstructed_tree"])
        self.assertEqual((self.dest / "source.txt").read_text(), "candidate\n")
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), before)
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")
        self.assertTrue((self.dest / ".git" / "sparkglm-atlas-source.json").is_file())

    def test_hash_tamper_refused_before_destination(self):
        (self.root / "change.patch").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.run_prepare()
        self.assertFalse(self.dest.exists())

    def test_existing_destination_preserved(self):
        self.dest.mkdir()
        (self.dest / "keep").write_text("keep")
        with self.assertRaisesRegex(ValueError, "destination"):
            self.run_prepare()
        self.assertEqual((self.dest / "keep").read_text(), "keep")

    def test_dirty_source_refused(self):
        (self.repo / "unrelated").write_text("keep")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.run_prepare()
        self.assertFalse(self.dest.exists())

    def test_traversal_and_missing_pin_refused(self):
        self.data["patch_chain"][0]["path"] = "../change.patch"
        self.save()
        with self.assertRaisesRegex(ValueError, "path"):
            self.run_prepare()
        self.assertFalse(self.dest.exists())
        self.data["patch_chain"][0]["path"] = "change.patch"
        self.data["upstream"]["revision"] = "main"
        self.save()
        with self.assertRaisesRegex(ValueError, "revision"):
            self.run_prepare()

    def test_wrong_result_tree_fails_and_retains_destination_for_inspection(self):
        self.data["reconstructed_tree"] = "1" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "tree"):
            self.run_prepare()
        self.assertTrue(self.dest.exists())
        self.assertFalse((self.dest / ".git" / "sparkglm-atlas-source.json").exists())

    def test_wrong_file_hash_fails_closed(self):
        self.data["changed_files_sha256"]["source.txt"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "file hash"):
            self.run_prepare()
        self.assertFalse((self.dest / ".git" / "sparkglm-atlas-source.json").exists())

    def test_real_manifest_and_patch_hashes(self):
        source = HERE.parent / "nvidia-day2-source.json"
        manifest, _ = r.validate_manifest(source)
        self.assertEqual(manifest["engine_revision"], "faf4e874b2d418b67a5bc4ec5793c450c9f27ebb")
        self.assertEqual(manifest["changed_file_count"], 95)
        self.assertEqual(len(manifest["patch_chain"]), 4)
        self.assertEqual(len(manifest["changed_files_sha256"]), 95)


    def test_measured_manifest_and_checkpoint_selection(self):
        measured, _ = r.validate_manifest(r.MANIFESTS["measured"])
        latest, _ = r.validate_manifest(r.MANIFESTS["latest"])
        self.assertEqual(r.DEFAULT_MANIFEST, r.MANIFESTS["latest"])
        self.assertEqual(measured["engine_revision"], "9b3e316a7f5ca92ec22a68414e7aefeda32023dd")
        self.assertEqual(measured["reconstructed_tree"], "72aec3455db6ab35a75985b3f390cb659d274244")
        self.assertEqual(measured["changed_file_count"], 81)
        self.assertEqual(measured["patch_chain"][:3], latest["patch_chain"][:3])
        self.assertNotEqual(measured["patch_chain"][3], latest["patch_chain"][3])
        bad = subprocess.run([sys.executable, str(HERE / "reconstruct.py"), "--checkpoint", "guessed"], capture_output=True)
        self.assertEqual(bad.returncode, 2)


class RealTreeTests(unittest.TestCase):
    def test_both_real_trees_in_private_temporary_indexes(self):
        if SOURCE_REPO is None:
            self.skipTest("--source-repo is required for local full-tree verification")
        source = pathlib.Path(SOURCE_REPO).resolve()
        objects = git(source, "rev-parse", "--path-format=absolute", "--git-path", "objects")
        before_head = git(source, "rev-parse", "HEAD")
        before_status = git(source, "status", "--porcelain", "--untracked-files=all")
        for checkpoint in ("measured", "latest"):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as name:
                temp = pathlib.Path(name)
                repo = temp / "objects-only.git"
                subprocess.run(["git", "init", "--bare", "-q", str(repo)], check=True)
                env = dict(r.ENV, GIT_ALTERNATE_OBJECT_DIRECTORIES=objects, GIT_INDEX_FILE=str(temp / "index"))
                def isolated(*args):
                    return subprocess.check_output(r.GIT + ["-C", str(repo), *args], env=env, stderr=subprocess.PIPE)
                data, patches = r.validate_manifest(r.MANIFESTS[checkpoint])
                isolated("read-tree", data["upstream"]["revision"])
                for patch in patches:
                    isolated("apply", "--cached", "--check", "--whitespace=nowarn", str(patch))
                    isolated("apply", "--cached", "--whitespace=nowarn", str(patch))
                tree = isolated("write-tree").decode().strip()
                self.assertEqual(tree, data["reconstructed_tree"])
                self.assertEqual(tree, git(source, "rev-parse", data["engine_revision"] + "^{tree}"))
                for path, expected in data["changed_files_sha256"].items():
                    payload = isolated("show", ":" + path)
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), expected, path)
        self.assertEqual(git(source, "rev-parse", "HEAD"), before_head)
        self.assertEqual(git(source, "status", "--porcelain", "--untracked-files=all"), before_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-repo", type=pathlib.Path)
    known, remaining = parser.parse_known_args()
    SOURCE_REPO = known.source_repo
    unittest.main(argv=[sys.argv[0]] + remaining)
