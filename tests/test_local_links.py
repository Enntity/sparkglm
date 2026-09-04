#!/usr/bin/env python3
"""Reject broken relative Markdown links in tracked project documentation."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    failures: list[str] = []
    checked = 0
    for relative in tracked:
        source = ROOT / relative
        for match in LINK.finditer(source.read_text(errors="replace")):
            raw = match.group(1).strip().split(maxsplit=1)[0].strip("<>")
            if not raw or raw.startswith(("#", "http://", "https://", "mailto:")):
                continue
            destination = unquote(raw.split("#", 1)[0])
            if not destination:
                continue
            checked += 1
            if not (source.parent / destination).resolve().exists():
                line = source.read_text(errors="replace").count("\n", 0, match.start()) + 1
                failures.append(f"{relative}:{line}: missing relative link {raw}")
    if failures:
        print("local Markdown link check: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"local Markdown link check: PASS ({checked} links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
