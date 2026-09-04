#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail closed on source drift from the retained posted-video image.

This is a source gate, not a binary-reproducibility or performance claim.
Only three explicit, byte-pinned nonfunctional differences are allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path, expected: dict, allowed: dict | None = None,
           inventory: bool = False) -> dict:
    allowed = allowed or {}
    failures = []
    exceptions = []
    for name, digest in expected.items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual == digest:
            continue
        exception = allowed.get(name, {})
        if actual is not None and actual == exception.get("sha256"):
            exceptions.append({"path": name, "reason": exception["reason"]})
        else:
            failures.append({"path": name, "expected": digest, "actual": actual})
    if inventory:
        observed = {str(p.relative_to(root)) for p in (root / "vllm").rglob("*.py")}
        failures.extend({"unexpected_python_file": p} for p in sorted(observed - expected.keys()))
    return {"passed": not failures, "files_checked": len(expected),
            "allowed_differences": exceptions, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="site-packages for python, extracted vLLM tree for native")
    parser.add_argument("--kind", choices=("python", "video-runtime", "native", "exl3"), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.kind in ("python", "video-runtime"):
        expected = dict(manifest["python_files"])
        if args.kind == "video-runtime":
            expected.update({p: item["sha256"] for p, item in
                             manifest["video_runtime_overrides"].items()})
        report = verify(args.root, expected,
                        manifest["allowed_nonfunctional_differences"], inventory=True)
        if args.kind == "video-runtime":
            report["runtime_transformations"] = manifest["video_runtime_overrides"]
    elif args.kind == "exl3":
        report = verify(args.root, manifest["exl3_compile_inputs"],
                        manifest["allowed_exl3_license_differences"])
    else:
        report = verify(args.root, manifest["native_source"])
    report.update(reference_image=manifest["reference_image"], kind=args.kind)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
