#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Require structured provenance on performance-sensitive commits."""

from __future__ import annotations

import re
import subprocess
import sys


RUNTIME_PATHS = (
    "Dockerfile",
    "ablit/",
    "overlay/",
    "patches/",
    "start.sh",
    "scripts/boot-shape-warmup.sh",
)
HEADINGS = ("Provenance:", "Original work:", "Verification:")
FULL_REVISION = re.compile(r"\b[0-9a-f]{40}\b")
URL = re.compile(r"https://[^\s>)]+")


def is_runtime_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in RUNTIME_PATHS)


def validate_message(message: str, paths: list[str]) -> list[str]:
    if not any(is_runtime_path(path) for path in paths):
        return []
    errors = [f"missing {heading}" for heading in HEADINGS if heading not in message]
    provenance = message.partition("Provenance:")[2].partition("Original work:")[0]
    if provenance and "original implementation" not in provenance.lower():
        if not URL.search(provenance):
            errors.append("external provenance needs a canonical https URL")
        if not FULL_REVISION.search(provenance):
            errors.append("external provenance needs a full 40-character revision")
    return errors


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_commit_provenance.py BASE_SHA HEAD_SHA", file=sys.stderr)
        return 2
    base, head = argv
    commits = git("rev-list", "--reverse", f"{base}..{head}").splitlines()
    failures = 0
    checked = 0
    for commit in commits:
        parents = git("show", "-s", "--format=%P", commit).split()
        if len(parents) > 1:
            continue
        paths = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
        if not any(is_runtime_path(path) for path in paths):
            continue
        checked += 1
        message = git("show", "-s", "--format=%B", commit)
        errors = validate_message(message, paths)
        if errors:
            failures += 1
            subject = message.splitlines()[0]
            print(f"{commit[:12]} {subject}", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
    if failures:
        return 1
    print(f"commit provenance: PASS ({checked} runtime commits checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
