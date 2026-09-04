#!/usr/bin/env python3
"""Unit tests for qualification record validation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "qualification.py"


def module():
    spec = importlib.util.spec_from_file_location("qualification", MODULE_PATH)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def record_for(bundle: Path, artifact: Path) -> dict:
    q = module()
    return {
        "schema": q.SCHEMA,
        "id": bundle.name,
        "title": "Example",
        "date": "2026-09-04",
        "state": "candidate",
        "qualification_level": "G0",
        "target": "test the verifier",
        "change": {
            "baseline_ref": None,
            "candidate_ref": "1" * 40,
            "working_tree_clean": True,
            "provenance": ["original"],
        },
        "environment": {},
        "gates": [{"id": "G0", "status": "pass", "summary": "unit test"}],
        "metrics": [],
        "artifacts": [
            {"path": artifact.name, "kind": "raw-receipt", "sha256": q.sha256(artifact)}
        ],
        "limitations": [],
        "attestation": {"status": "unattested", "maintainer": None, "date": None},
    }


def test_valid_record_and_checksum() -> None:
    q = module()
    with tempfile.TemporaryDirectory() as raw:
        bundle = Path(raw) / "example"
        bundle.mkdir()
        artifact = bundle / "raw.json"
        artifact.write_text("{}\n")
        record = bundle / "qualification.json"
        record.write_text(json.dumps(record_for(bundle, artifact)))
        assert q.verify_record(record) == []
        artifact.write_text("changed\n")
        assert any("checksum mismatch" in item for item in q.verify_record(record))


def test_paths_cannot_escape_bundle() -> None:
    q = module()
    with tempfile.TemporaryDirectory() as raw:
        bundle = Path(raw) / "example"
        bundle.mkdir()
        artifact = bundle / "raw.json"
        artifact.write_text("{}\n")
        payload = record_for(bundle, artifact)
        payload["artifacts"][0]["path"] = "../raw.json"
        record = bundle / "qualification.json"
        record.write_text(json.dumps(payload))
        assert any("escapes bundle" in item for item in q.verify_record(record))


def test_every_bundle_artifact_must_be_checksummed() -> None:
    q = module()
    with tempfile.TemporaryDirectory() as raw:
        bundle = Path(raw) / "example"
        bundle.mkdir()
        artifact = bundle / "raw.json"
        artifact.write_text("{}\n")
        (bundle / "forgotten.log").write_text("not in manifest\n")
        record = bundle / "qualification.json"
        record.write_text(json.dumps(record_for(bundle, artifact)))
        assert any("unchecksummed bundle artifact" in item for item in q.verify_record(record))


def test_checksum_discovers_new_artifact() -> None:
    q = module()
    with tempfile.TemporaryDirectory() as raw:
        bundle = Path(raw) / "example"
        bundle.mkdir()
        artifact = bundle / "raw.json"
        artifact.write_text("{}\n")
        extra = bundle / "notes.md"
        extra.write_text("notes\n")
        record = bundle / "qualification.json"
        record.write_text(json.dumps(record_for(bundle, artifact)))
        q.command_index = lambda: 0
        q.command_checksum(record)
        payload = json.loads(record.read_text())
        assert [entry["path"] for entry in payload["artifacts"]] == [
            "notes.md", "raw.json"
        ]
        assert q.verify_record(record) == []


def test_reviewed_record_requires_clean_commit_and_completed_gates() -> None:
    q = module()
    with tempfile.TemporaryDirectory() as raw:
        bundle = Path(raw) / "example"
        bundle.mkdir()
        artifact = bundle / "raw.json"
        artifact.write_text("{}\n")
        payload = record_for(bundle, artifact)
        payload["qualification_level"] = "G2"
        payload["change"]["working_tree_clean"] = False
        payload["gates"] = [
            {"id": "G0", "status": "pass", "summary": "ok"},
            {"id": "G1", "status": "not-run", "summary": ""},
            {"id": "G2", "status": "pass", "summary": "ok"},
        ]
        payload["attestation"] = {
            "status": "maintainer-reviewed",
            "maintainer": "maintainer",
            "date": "2026-09-04",
        }
        record = bundle / "qualification.json"
        record.write_text(json.dumps(payload))
        failures = q.verify_record(record)
        assert any("clean worktree" in item for item in failures)
        assert any("requires G1=pass" in item for item in failures)


def test_legacy_evidence_cannot_be_retroactively_reviewed() -> None:
    q = module()
    with tempfile.TemporaryDirectory() as raw:
        bundle = Path(raw) / "example"
        bundle.mkdir()
        artifact = bundle / "raw.json"
        artifact.write_text("{}\n")
        payload = record_for(bundle, artifact)
        payload["qualification_level"] = "legacy"
        payload["attestation"]["status"] = "maintainer-reviewed"
        record = bundle / "qualification.json"
        record.write_text(json.dumps(payload))
        failures = q.verify_record(record)
        assert any("legacy record must use legacy attestation" in item for item in failures)
        assert any("legacy evidence cannot be certified" in item for item in failures)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"qualification record tests OK ({len(tests)} tests)")
