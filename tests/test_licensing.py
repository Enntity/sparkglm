#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify source pins, retained notices, and file-license boundaries."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "upstreams.json"
ATTRIBUTION = ROOT / "docs" / "ATTRIBUTION.md"
FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
SPDX = re.compile(
    r"(?m)^(?:#|//|<!--)\s*SPDX-License-Identifier:\s*([^\r\n]+?)\s*(?:-->)?$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def tracked_files() -> list[str]:
    return subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()


def matches(files: list[str], patterns: list[str]) -> set[str]:
    return {
        path for path in files
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
    }


def resolved_expression(path: str, manifest: dict) -> str:
    for rule in manifest["path_rules"]:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule["patterns"]):
            return rule["license_expression"]
    return manifest["default_license_expression"]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    files = tracked_files()
    attribution = ATTRIBUTION.read_text()
    assert manifest["schema"] == "sparkglm.provenance/v1"

    document_ids: set[str] = set()
    for document in manifest["license_documents"] + manifest["notices"]:
        assert document["id"] not in document_ids
        document_ids.add(document["id"])
        path = ROOT / document["path"]
        assert path.is_file(), document["path"]
        assert sha256(path) == document["sha256"], document["path"]
        if document.get("upstream_git_blob"):
            assert git_blob(path) == document["upstream_git_blob"], document["path"]

    source_ids: set[str] = set()
    for source in manifest["sources"]:
        assert source["id"] not in source_ids
        source_ids.add(source["id"])
        assert source["repository"].startswith("https://")
        assert FULL_REVISION.fullmatch(source["revision"]), source["id"]
        assert source["repository"] in attribution, source["id"]
        assert source["revision"] in attribution, source["id"]
        assert source["relationship"], source["id"]
        assert matches(files, source["covered_paths"]), source["id"]
        for notice_id in source["notice_ids"]:
            assert notice_id in document_ids, (source["id"], notice_id)

    for rule in manifest["path_rules"]:
        assert rule["patterns"] and rule["reason"]
        assert matches(files, rule["patterns"]), rule["patterns"]

    licensing = (ROOT / "docs" / "LICENSING.md").read_text()
    quant = (ROOT / 'docs/QUANT_ATTRIBUTION.md').read_text()
    notice = next(line[2:] for line in quant.splitlines() if line.startswith('> This work includes'))
    for relative in ('README.md', 'NOTICE'):
        assert notice in (ROOT / relative).read_text(), relative
    for phrase in ('Brandon M. Music', 'https://github.com/brandonmmusic-max/shapleymcg',
                   '@misc{music2026shapleymcg', '25a44fdbf16862a46b7cc9921142c6c81350af2f'):
        assert phrase in quant, phrase
    for relative in ('docs/RESULTS.md', 'results/README.md', 'docs/ATTRIBUTION.md'):
        assert 'QUANT_ATTRIBUTION.md' in (ROOT / relative).read_text(), relative
    assert 'source-available' in licensing and 'named-party' in licensing
    assert 'do **not** remove' in licensing
    for relative in ('README.md', 'NOTICE', 'docs/LICENSING.md'):
        assert 'benchmarks/staggered_openai.py' in (ROOT / relative).read_text(), relative
    for artifact in manifest["non_redistributed_artifacts"]:
        assert FULL_REVISION.fullmatch(artifact["revision"]), artifact["id"]
        assert artifact["revision"] in attribution, artifact["id"]
        assert artifact["license"] in licensing, artifact["id"]

    known_spdx = {
        "Apache-2.0", "MIT", "AGPL-3.0-only", "MIT AND Apache-2.0"
    }
    spdx_files = 0
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            continue
        head = path.read_bytes()[:2048].decode("utf-8", errors="ignore")
        match = SPDX.search(head)
        if not match:
            continue
        spdx_files += 1
        identifier = match.group(1)
        assert identifier in known_spdx, (relative, identifier)
        expression = resolved_expression(relative, manifest)
        assert identifier == expression or identifier in expression, (
            relative, identifier, expression
        )

    assert resolved_expression("files/chat_template.jinja", manifest) == "LicenseRef-GLM-5.3"
    assert resolved_expression("research/atlas/bench/native_nccl_bench.py", manifest) == "AGPL-3.0-only"
    assert resolved_expression("benchmarks/staggered_openai.py", manifest) == "AGPL-3.0-only"
    assert resolved_expression(".github/FUNDING.yml", manifest) == "MIT"
    assert spdx_files >= 25

    start = (ROOT / "start.sh").read_text()
    env = (ROOT / ".env.example").read_text()
    for name, revision in (
        ("MODEL_REVISION", "25a44fdbf16862a46b7cc9921142c6c81350af2f"),
        ("MODEL_FALLBACK_REVISION", "5ab363a8dcf6405955fd5f99671e01a1c9fb124b"),
        ("DFLASH_REVISION", "7d74cdd881ed7e32c31175984a67823127b66cfe"),
    ):
        assert f'{name}="${{{name}:-{revision}}}"' in start
        assert f"{name}={revision}" in env
    assert 'count_revision_shards "$MODEL_PATH" "$MODEL_REVISION"' in start
    assert 'count_revision_shards "$FALLBACK_MODEL_PATH" "$MODEL_FALLBACK_REVISION"' in start
    assert 'find "$DFLASH_PATH/snapshots/$DFLASH_REVISION"' in start
    assert 'hf_download_repo "$DFLASH_MODEL"' in start

    print(
        f"licensing and attribution: PASS ({len(manifest['sources'])} sources, "
        f"{len(manifest['path_rules'])} boundary rules, {spdx_files} SPDX files)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
