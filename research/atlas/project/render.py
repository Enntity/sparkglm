#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Render the frozen experiment's Docker argv; never execute it."""
import argparse
import json
import hashlib
from pathlib import Path
import shlex
from string import Template

HERE = Path(__file__).resolve().parent
PARAMETERS = ("rank", "master", "bind", "port", "nic", "hca", "name", "image",
              "overlay", "original", "cuda_cache", "template", "flash_library_dir", "native_library")

def render(values):
    if set(values) != set(PARAMETERS):
        raise ValueError("supply exactly the documented site parameters")
    for key in PARAMETERS:
        if not values.get(key) or any(c in str(values[key]) for c in "\r\n\x00"):
            raise ValueError(f"missing or invalid parameter: {key}")
    if str(values["rank"]) not in ("0", "1"):
        raise ValueError("rank must be 0 or 1")
    if not str(values["port"]).isdigit() or not 1 <= int(values["port"]) <= 65535:
        raise ValueError("port must be 1..65535")
    for key in ("overlay", "original", "cuda_cache", "template", "flash_library_dir", "native_library"):
        if not Path(values[key]).is_absolute() or "," in values[key]:
            raise ValueError(f"{key} must be an absolute path without commas")
    profile = json.loads((HERE / "profile.json").read_text())
    def expand(items):
        return [Template(item).substitute(values) for item in items]
    argv = ["docker", "run", "-d"] + expand(profile["docker_options"])
    for mount in expand(profile["mounts"] + profile["artifact_mounts"]):
        argv.extend(["--mount", mount])
    argv.extend(["--entrypoint=/usr/bin/env", values["image"], "-i"])
    argv.extend(expand(profile["environment"]))
    argv.extend(["/usr/local/bin/spark"] + expand(profile["server_argv"]))
    return argv


def verify_local_artifacts(values):
    """Explicit operator preflight on the machine owning these paths; never execute Docker."""
    render(values)
    profile = json.loads((HERE / "profile.json").read_text())
    artifacts = [(Path(values["template"]), profile["template_sha256"])]
    for spec in profile["external_artifacts"]:
        path = Path(values[spec["parameter"]])
        if spec["filename"]:
            path = path / spec["filename"]
        artifacts.append((path, spec["sha256"]))
    for path, expected in artifacts:
        try:
            with path.open("rb") as stream:
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
                digest = hasher.hexdigest()
        except OSError as error:
            raise ValueError(f"required artifact unavailable: {path}") from error
        if digest != expected:
            raise ValueError(f"required artifact SHA-256 mismatch: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in PARAMETERS:
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    parser.add_argument("--verify-local-artifacts", action="store_true", help="hash the template and all three libraries before rendering")
    args = vars(parser.parse_args())
    output_format = args.pop("format")
    verify = args.pop("verify_local_artifacts")
    try:
        if verify:
            verify_local_artifacts(args)
        argv = render(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(argv, indent=2) if output_format == "json" else shlex.join(argv))
