# SPDX-License-Identifier: AGPL-3.0-only
"""Reconstruct pinned Atlas source locally; never contacts a remote or starts serving."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

MANIFESTS = {"latest": Path(__file__).resolve().parent.parent / "nvidia-day2-source.json",
             "measured": Path(__file__).resolve().parent.parent / "nvidia-r23-source.json"}
DEFAULT_MANIFEST = MANIFESTS["latest"]
ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
GIT = ["git", "-c", "filter.lfs.required=false", "-c", "filter.lfs.smudge=",
       "-c", "filter.lfs.process=", "-c", "filter.lfs.clean=", "-c", "submodule.recurse=false"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    require(isinstance(value, str) and value and "\\" not in value, "invalid path")
    path = Path(value)
    require(not path.is_absolute() and all(p not in ("..", ".git") for p in path.parts),
            "unsafe relative path")
    return path


def git(repo, *args):
    return subprocess.check_output(GIT + ["-C", str(repo), *args], env=ENV,
                                   text=True, stderr=subprocess.PIPE).strip()


def validate_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    require(data.get("schema") == "sparkglm.atlas-source-export/v2", "unsupported manifest schema")
    for label, value in [("upstream revision", data["upstream"]["revision"]),
                         ("engine revision", data["engine_revision"]),
                         ("tree", data["reconstructed_tree"])]:
        require(isinstance(value, str) and re.fullmatch("[0-9a-f]{40}", value),
                "invalid " + label)
    chain = data["patch_chain"]
    require(isinstance(chain, list) and chain, "empty patch chain")
    seen = set()
    patches = []
    for entry in chain:
        name = relative_path(entry["path"])
        require(str(name) not in seen, "duplicate patch path")
        seen.add(str(name))
        file = path.parent / name
        require(not file.is_symlink() and file.resolve().is_relative_to(path.parent),
                "patch path escapes manifest directory")
        payload = file.read_bytes()
        require(sha(payload) == entry["sha256"], "patch hash mismatch: " + str(name))
        require(b"GIT binary patch" not in payload and b"Binary files " not in payload,
                "binary patches prohibited")
        patches.append(file)
    files = data["changed_files_sha256"]
    require(isinstance(files, dict) and len(files) == data["changed_file_count"],
            "changed file count mismatch")
    for name, digest in files.items():
        relative_path(name)
        require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
                "invalid file hash")
    return data, patches


def prepare(manifest_path, source_repo, destination):
    data, patches = validate_manifest(manifest_path)
    raw_dest = Path(destination).expanduser()
    require(not raw_dest.exists() and not raw_dest.is_symlink(),
            "destination must be fresh and absent; existing data is never removed")
    dest = raw_dest.resolve()
    source = Path(source_repo).expanduser().resolve()
    require(source.is_dir(), "upstream source repository is required")
    require(git(source, "rev-parse", "--is-inside-work-tree") == "true",
            "source must be a local non-bare Git checkout")
    require(git(source, "status", "--porcelain", "--untracked-files=all") == "",
            "source checkout must be clean, including untracked files")
    base = data["upstream"]["revision"]
    require(git(source, "rev-parse", base + "^{commit}") == base,
            "pinned upstream commit unavailable locally")
    require(dest.parent.is_dir(), "destination parent directory must already exist")
    # A separate clone avoids changing the source index/worktree or creating a worktree.
    subprocess.run(GIT + ["clone", "--local", "--no-hardlinks", "--dissociate",
                         "--no-checkout", "--", str(source), str(dest)],
                   env=ENV, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    git(dest, "checkout", "--detach", base)
    require(git(dest, "rev-parse", "HEAD") == base, "destination is not exact upstream base")
    require(git(dest, "status", "--porcelain", "--untracked-files=all") == "",
            "fresh destination is not clean before patch application")
    for file in patches:
        # Revalidate against the manifest immediately before applying, avoiding stale reads.
        expected = next(e["sha256"] for e in data["patch_chain"]
                        if (Path(manifest_path).resolve().parent / e["path"]) == file)
        require(sha(file.read_bytes()) == expected, "patch hash changed during preparation")
        git(dest, "apply", "--index", "--whitespace=nowarn", "--check", str(file))
        git(dest, "apply", "--index", "--whitespace=nowarn", str(file))
    tree = git(dest, "write-tree")
    require(tree == data["reconstructed_tree"], "reconstructed tree mismatch")
    for name, expected in data["changed_files_sha256"].items():
        file = dest / name
        require(not file.is_symlink() and file.resolve().is_relative_to(dest),
                "changed source file escapes destination")
        require(sha(file.read_bytes()) == expected, "file hash mismatch: " + name)
    require(git(dest, "diff", "--name-only") == "", "unstaged reconstruction mismatch")
    receipt = {"schema": "sparkglm.atlas-reconstruction/v1", "engine_revision": data["engine_revision"],
               "tree": tree, "upstream_revision": base,
               "manifest_sha256": sha(Path(manifest_path).read_bytes()),
               "patch_chain": data["patch_chain"],
               "note": "Source reconstructed; external native build dependencies remain separate. No serving started."}
    (dest / ".git" / "sparkglm-atlas-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=("latest", "measured"), default="latest",
                        help="latest diagnostic source (default) or exact measured r23 source")
    parser.add_argument("--source-repo", required=True, type=Path,
                        help="clean local Mango-compatible Git checkout containing the pinned base")
    parser.add_argument("--destination", required=True, type=Path,
                        help="fresh absent destination; failures are retained for inspection")
    args = parser.parse_args()
    try:
        receipt = prepare(MANIFESTS[args.checkpoint], args.source_repo, args.destination)
        receipt["checkpoint"] = args.checkpoint
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, "Reconstruction refused: " + str(error) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
