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
import re
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


def candidate_expectations(manifest: dict, declaration: dict, kind: str,
                           expected: dict, allowed: dict) -> tuple[dict, dict]:
    """Permit only explicitly declared hashes, without editing the reference."""
    if declaration.get("schema") != "sparkglm.candidate-sources/v1":
        raise ValueError("invalid candidate source declaration schema")
    groups = declaration.get("changes")
    if not isinstance(groups, dict) or set(groups) - {"python", "native", "exl3"}:
        raise ValueError("candidate changes must be python/native/exl3 maps")
    sources = {"python": "python_files", "native": "native_source",
               "exl3": "exl3_compile_inputs"}
    exceptions = {"python": "allowed_nonfunctional_differences",
                  "native": "", "exl3": "allowed_exl3_license_differences"}
    # Validate every group, even when this invocation checks only one stage.
    for group, changes in groups.items():
        if not isinstance(changes, dict):
            raise ValueError("each candidate group must be a map")
        for name, change in changes.items():
            if (not isinstance(change, dict) or not isinstance(name, str)
                    or name.startswith("/") or ".." in name.split("/")):
                raise ValueError("invalid candidate path or declaration")
            original = manifest[sources[group]].get(name)
            alternate = manifest.get(exceptions[group], {}).get(name, {}).get("sha256")
            if original is None or change.get("reference_sha256") not in {original, alternate} - {None}:
                raise ValueError(f"{group}/{name}: reference hash is not in the frozen manifest")
            if not re.fullmatch(r"[0-9a-f]{64}", str(change.get("candidate_sha256", ""))):
                raise ValueError(f"{group}/{name}: candidate SHA-256 required")
            if not isinstance(change.get("reason"), str) or not change["reason"].strip():
                raise ValueError(f"{group}/{name}: reason required")
    expected, allowed = dict(expected), dict(allowed)
    for name, change in groups.get(kind, {}).items():
        expected[name] = change["candidate_sha256"]
        # A candidate must match its declared bytes, not an older exception.
        allowed.pop(name, None)
    return expected, allowed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="site-packages for python, extracted vLLM tree for native")
    parser.add_argument("--kind", choices=("python", "video-runtime", "native", "exl3"), required=True)
    parser.add_argument("--candidate-manifest", type=Path,
                        help="explicit experimental source hashes; never source parity certification")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.candidate_manifest and args.kind == "video-runtime":
        parser.error("video-runtime remains an immutable reference-only check")
    if args.kind in ("python", "video-runtime"):
        expected = dict(manifest["python_files"])
        if args.kind == "video-runtime":
            expected.update({p: item["sha256"] for p, item in
                             manifest["video_runtime_overrides"].items()})
        allowed = manifest["allowed_nonfunctional_differences"]
        inventory = True
    elif args.kind == "exl3":
        expected = manifest["exl3_compile_inputs"]
        allowed = manifest["allowed_exl3_license_differences"]
        inventory = False
    else:
        expected, allowed, inventory = manifest["native_source"], {}, False
    declaration = None
    if args.candidate_manifest:
        declaration = json.loads(args.candidate_manifest.read_text())
        try:
            expected, allowed = candidate_expectations(manifest, declaration, args.kind, expected, allowed)
        except ValueError as exc:
            parser.error(str(exc))
    report = verify(args.root, expected, allowed, inventory=inventory)
    report["source_profile"] = "candidate" if declaration is not None else "reference"
    if declaration is not None:
        report["declared_changes"] = declaration["changes"].get(args.kind, {})
        report["qualification"] = "unqualified experiment; not video source parity"
    if args.kind == "video-runtime":
        report["runtime_transformations"] = manifest["video_runtime_overrides"]
    report.update(reference_image=manifest["reference_image"], kind=args.kind)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
