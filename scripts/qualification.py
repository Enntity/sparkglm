#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create and verify small, checksum-bound SparkGLM qualification bundles."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INDEX = RESULTS / "index.json"
SCHEMA = "sparkglm.qualification/v1"
INDEX_SCHEMA = "sparkglm.results-index/v1"
STATES = {
    "experiment", "candidate", "accepted", "promoted", "rejected",
    "baseline", "measurement",
}
LEVELS = ("legacy", "G0", "G1", "G2", "G3", "G4", "G5")
GATE_STATUS = {"pass", "fail", "not-run", "not-applicable"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def load_record(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read qualification JSON: {exc}") from exc


def _relative_artifact(bundle: Path, raw: object) -> tuple[Path | None, str | None]:
    if not isinstance(raw, str) or not raw:
        return None, "artifact path must be a non-empty string"
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None, f"artifact path escapes bundle: {raw}"
    resolved = (bundle / candidate).resolve()
    try:
        resolved.relative_to(bundle.resolve())
    except ValueError:
        return None, f"artifact path escapes bundle: {raw}"
    return resolved, None


def verify_record(path: Path) -> list[str]:
    failures: list[str] = []
    try:
        record = load_record(path)
    except ValueError as exc:
        return [str(exc)]
    bundle = path.parent
    required = (
        "schema", "id", "title", "date", "state", "qualification_level",
        "target", "change", "environment", "gates", "metrics", "artifacts",
        "limitations", "attestation",
    )
    for key in required:
        if key not in record:
            failures.append(f"{path}: missing {key}")
    if failures:
        return failures
    if record["schema"] != SCHEMA:
        failures.append(f"{path}: unsupported schema {record['schema']!r}")
    if not isinstance(record["id"], str) or not IDENTIFIER.fullmatch(record["id"]):
        failures.append(f"{path}: invalid id")
    if record["id"] != bundle.name:
        failures.append(f"{path}: id must match bundle directory")
    if not isinstance(record["title"], str) or not record["title"].strip():
        failures.append(f"{path}: title must be non-empty")
    if record["state"] not in STATES:
        failures.append(f"{path}: invalid state {record['state']!r}")
    if record["qualification_level"] not in LEVELS:
        failures.append(f"{path}: invalid qualification level")
    if not isinstance(record["target"], str) or not record["target"].strip():
        failures.append(f"{path}: target must be non-empty")
    for key in ("environment",):
        if not isinstance(record[key], dict):
            failures.append(f"{path}: {key} must be an object")
    for key in ("gates", "metrics", "artifacts", "limitations"):
        if not isinstance(record[key], list):
            failures.append(f"{path}: {key} must be an array")

    change = record["change"]
    if not isinstance(change, dict):
        failures.append(f"{path}: change must be an object")
    else:
        for key in ("baseline_ref", "candidate_ref", "working_tree_clean", "provenance"):
            if key not in change:
                failures.append(f"{path}: change missing {key}")
        if not isinstance(change.get("provenance"), list):
            failures.append(f"{path}: change.provenance must be an array")

    seen_gates: set[str] = set()
    for gate in record["gates"] if isinstance(record["gates"], list) else []:
        if not isinstance(gate, dict):
            failures.append(f"{path}: gate must be an object")
            continue
        gate_id = gate.get("id")
        status = gate.get("status")
        if gate_id not in LEVELS[1:]:
            failures.append(f"{path}: invalid gate id {gate_id!r}")
        elif gate_id in seen_gates:
            failures.append(f"{path}: duplicate gate {gate_id}")
        seen_gates.add(gate_id)
        if status not in GATE_STATUS:
            failures.append(f"{path}: invalid status for gate {gate_id}")
        if not isinstance(gate.get("summary", ""), str):
            failures.append(f"{path}: gate {gate_id} summary must be a string")

    listed_artifacts: set[str] = set()
    for artifact in record["artifacts"] if isinstance(record["artifacts"], list) else []:
        if not isinstance(artifact, dict):
            failures.append(f"{path}: artifact must be an object")
            continue
        target, error = _relative_artifact(bundle, artifact.get("path"))
        if error:
            failures.append(f"{path}: {error}")
            continue
        assert target is not None
        artifact_path = artifact.get("path")
        if artifact_path in listed_artifacts:
            failures.append(f"{path}: duplicate artifact {artifact_path}")
        listed_artifacts.add(artifact_path)
        if not target.is_file():
            failures.append(f"{path}: missing artifact {artifact.get('path')}")
            continue
        expected = artifact.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            failures.append(f"{path}: invalid checksum for {artifact.get('path')}")
        elif sha256(target) != expected:
            failures.append(f"{path}: checksum mismatch for {artifact.get('path')}")
        if not isinstance(artifact.get("kind"), str) or not artifact["kind"]:
            failures.append(f"{path}: artifact kind must be non-empty")

    present_artifacts = {
        candidate.relative_to(bundle).as_posix()
        for candidate in bundle.rglob("*")
        if candidate.is_file() and candidate != path
    }
    for unlisted in sorted(present_artifacts - listed_artifacts):
        failures.append(f"{path}: unchecksummed bundle artifact {unlisted}")
    for absent in sorted(listed_artifacts - present_artifacts):
        failures.append(f"{path}: listed artifact is absent {absent}")

    attestation = record["attestation"]
    if not isinstance(attestation, dict):
        failures.append(f"{path}: attestation must be an object")
    else:
        attestation_status = attestation.get("status")
        if attestation_status not in {"unattested", "legacy", "maintainer-reviewed"}:
            failures.append(f"{path}: invalid attestation status")
        if record["qualification_level"] == "legacy" and attestation_status != "legacy":
            failures.append(f"{path}: legacy record must use legacy attestation")
        if attestation_status == "maintainer-reviewed":
            if record["qualification_level"] == "legacy":
                failures.append(f"{path}: legacy evidence cannot be certified")
            if not isinstance(change, dict) or change.get("working_tree_clean") is not True:
                failures.append(f"{path}: reviewed qualification requires a clean worktree")
            if not isinstance(change, dict) or not HEX40.fullmatch(str(change.get("candidate_ref", ""))):
                failures.append(f"{path}: reviewed qualification requires a full candidate SHA")
            if not attestation.get("maintainer") or not attestation.get("date"):
                failures.append(f"{path}: reviewed qualification requires maintainer and date")
            target_level = LEVELS.index(record["qualification_level"])
            gate_map = {
                gate.get("id"): gate.get("status")
                for gate in record["gates"]
                if isinstance(gate, dict)
            }
            for gate_id in LEVELS[1:target_level + 1]:
                if gate_map.get(gate_id) != "pass":
                    failures.append(
                        f"{path}: reviewed {record['qualification_level']} requires {gate_id}=pass"
                    )
            if target_level >= LEVELS.index("G3"):
                environment = record["environment"]
                for field in (
                    "hardware", "topology", "image_digests", "model_revision",
                    "configuration",
                ):
                    if not environment.get(field):
                        failures.append(
                            f"{path}: reviewed G3+ qualification requires environment.{field}"
                        )
                if not record["artifacts"]:
                    failures.append(f"{path}: reviewed G3+ qualification requires raw artifacts")
                gate_map = {
                    gate.get("id"): gate.get("status")
                    for gate in record["gates"]
                    if isinstance(gate, dict)
                }
                if gate_map.get("G4") != "pass" and not record["limitations"]:
                    failures.append(
                        f"{path}: G3 without G4 requires an explicit quality limitation"
                    )
        if (
            record["qualification_level"] != "legacy"
            and record["state"] in {"accepted", "promoted"}
            and attestation_status != "maintainer-reviewed"
        ):
            failures.append(
                f"{path}: post-policy {record['state']} record requires maintainer review"
            )
    return failures


def record_paths() -> list[Path]:
    return sorted(RESULTS.glob("**/qualification.json"))


def index_payload() -> dict:
    records = []
    for path in record_paths():
        record = load_record(path)
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "id": record.get("id"),
            "title": record.get("title"),
            "date": record.get("date"),
            "state": record.get("state"),
            "qualification_level": record.get("qualification_level"),
            "target": record.get("target"),
        })
    return {"schema": INDEX_SCHEMA, "records": records}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def command_index() -> int:
    write_json(INDEX, index_payload())
    print(f"wrote {INDEX.relative_to(ROOT)}")
    return 0


def command_verify_all() -> int:
    failures: list[str] = []
    paths = record_paths()
    if not paths:
        failures.append("no qualification records found")
    for path in paths:
        failures.extend(verify_record(path))
    if not INDEX.is_file():
        failures.append("results/index.json is missing; run qualification.py index")
    else:
        try:
            actual_index = json.loads(INDEX.read_text())
        except json.JSONDecodeError as exc:
            failures.append(f"results/index.json is invalid: {exc}")
        else:
            if actual_index != index_payload():
                failures.append("results/index.json is stale; run qualification.py index")
    if failures:
        print("qualification verification: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"qualification verification: PASS ({len(paths)} records)")
    return 0


def new_record(args: argparse.Namespace) -> int:
    if not IDENTIFIER.fullmatch(args.id):
        raise SystemExit("--id must use lowercase letters, digits, dot, dash, or underscore")
    bundle = RESULTS / "candidates" / args.id
    if bundle.exists():
        raise SystemExit(f"bundle already exists: {bundle}")
    bundle.mkdir(parents=True)
    (bundle / "raw").mkdir()
    head = git("rev-parse", "HEAD")
    clean = not bool(git("status", "--porcelain"))
    report = bundle / "RESULT.md"
    report.write_text(
        f"# {args.title}\n\n## Hypothesis\n\n## Frozen configuration\n\n"
        "## Results\n\n## Correctness\n\n## Limitations\n"
    )
    record = {
        "schema": SCHEMA,
        "id": args.id,
        "title": args.title,
        "date": dt.date.today().isoformat(),
        "state": "candidate",
        "qualification_level": "G0",
        "target": args.target,
        "change": {
            "baseline_ref": args.baseline_ref,
            "candidate_ref": head,
            "working_tree_clean": clean,
            "provenance": [],
        },
        "environment": {
            "capture_platform": platform.platform(),
            "python": platform.python_version(),
            "image_digests": [],
            "model_revision": None,
            "quant_revision": None,
            "drafter_revision": None,
            "configuration": {},
        },
        "gates": [
            {"id": gate, "status": "not-run", "summary": ""}
            for gate in LEVELS[1:]
        ],
        "metrics": [],
        "artifacts": [
            {"path": "RESULT.md", "kind": "human-report", "sha256": sha256(report)}
        ],
        "limitations": [],
        "attestation": {"status": "unattested", "maintainer": None, "date": None},
    }
    write_json(bundle / "qualification.json", record)
    command_index()
    print(f"created {bundle.relative_to(ROOT)} (clean={clean})")
    return 0


def command_checksum(path: Path) -> int:
    record = load_record(path)
    bundle = path.parent
    artifacts = record.setdefault("artifacts", [])
    listed = {
        artifact.get("path")
        for artifact in artifacts
        if isinstance(artifact, dict)
    }
    for target in sorted(bundle.rglob("*")):
        if not target.is_file() or target == path:
            continue
        relative = target.relative_to(bundle).as_posix()
        if relative in listed:
            continue
        suffix = target.suffix.lower()
        if suffix == ".md":
            kind = "human-report"
        elif suffix == ".json":
            kind = "raw-receipt"
        else:
            kind = "raw-evidence"
        artifacts.append({"path": relative, "kind": kind, "sha256": ""})
        listed.add(relative)
    for artifact in artifacts:
        target, error = _relative_artifact(bundle, artifact.get("path"))
        if error or target is None or not target.is_file():
            raise SystemExit(error or f"missing artifact: {artifact.get('path')}")
        artifact["sha256"] = sha256(target)
    artifacts.sort(key=lambda artifact: artifact["path"])
    write_json(path, record)
    command_index()
    try:
        display_path = path.relative_to(ROOT)
    except ValueError:
        display_path = path
    print(f"updated checksums in {display_path}")
    return 0


def legacy_record(args: argparse.Namespace) -> int:
    if not IDENTIFIER.fullmatch(args.id):
        raise SystemExit("invalid legacy --id")
    bundle = RESULTS / "legacy" / args.id
    if bundle.exists():
        raise SystemExit(f"bundle already exists: {bundle}")
    bundle.mkdir(parents=True)
    artifacts = []
    for source_arg in args.files:
        source = (ROOT / source_arg).resolve()
        if not source.is_file():
            raise SystemExit(f"missing legacy artifact: {source_arg}")
        destination_name = source.name
        destination = bundle / destination_name
        if destination.exists():
            raise SystemExit(f"duplicate legacy artifact name: {destination_name}")
        shutil.move(str(source), destination)
        kind = "human-report" if destination.suffix.lower() == ".md" else "raw-receipt"
        artifacts.append({
            "path": destination.name,
            "kind": kind,
            "sha256": sha256(destination),
        })
    record = {
        "schema": SCHEMA,
        "id": args.id,
        "title": args.title,
        "date": args.date,
        "state": args.state,
        "qualification_level": "legacy",
        "target": args.target,
        "change": {
            "baseline_ref": None,
            "candidate_ref": None,
            "working_tree_clean": None,
            "provenance": ["Migrated from pre-policy SparkGLM evidence without retroactive certification."],
        },
        "environment": {},
        "gates": [],
        "metrics": [],
        "artifacts": artifacts,
        "limitations": [
            "This evidence predates qualification-v1.",
            "Only facts present in the retained artifact may be claimed.",
            "Missing commit, image, environment, warmup, or quality fields were not invented during migration.",
        ],
        "attestation": {"status": "legacy", "maintainer": None, "date": None},
    }
    write_json(bundle / "qualification.json", record)
    print(f"migrated {len(artifacts)} artifact(s) to {bundle.relative_to(ROOT)}")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-all")
    commands.add_parser("index")
    new = commands.add_parser("new")
    new.add_argument("--id", required=True)
    new.add_argument("--title", required=True)
    new.add_argument("--target", required=True)
    new.add_argument("--baseline-ref", required=True)
    checksum = commands.add_parser("checksum")
    checksum.add_argument("record", type=Path)
    legacy = commands.add_parser("legacy")
    legacy.add_argument("--id", required=True)
    legacy.add_argument("--title", required=True)
    legacy.add_argument("--date")
    legacy.add_argument("--state", choices=sorted(STATES), default="measurement")
    legacy.add_argument(
        "--target", default="Historical experiment evidence retained for reference"
    )
    legacy.add_argument("files", nargs="+")
    return top


def main() -> int:
    args = parser().parse_args()
    if args.command == "verify-all":
        return command_verify_all()
    if args.command == "index":
        return command_index()
    if args.command == "new":
        return new_record(args)
    if args.command == "checksum":
        path = args.record if args.record.is_absolute() else ROOT / args.record
        return command_checksum(path)
    if args.command == "legacy":
        return legacy_record(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
