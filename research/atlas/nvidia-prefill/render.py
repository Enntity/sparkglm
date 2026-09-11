#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Render the frozen experiment's Docker argv; never execute it."""
import argparse
import json
from pathlib import Path
import shlex
from string import Template

HERE = Path(__file__).resolve().parent
PARAMETERS = ("rank", "master", "bind", "port", "nic", "hca", "name", "image",
              "overlay", "original", "cuda_cache", "template")

def render(values):
    for key in PARAMETERS:
        if not values.get(key) or any(c in str(values[key]) for c in "\r\n\x00"):
            raise ValueError(f"missing or invalid parameter: {key}")
    if str(values["rank"]) not in ("0", "1"):
        raise ValueError("rank must be 0 or 1")
    if not str(values["port"]).isdigit() or not 1 <= int(values["port"]) <= 65535:
        raise ValueError("port must be 1..65535")
    for key in ("overlay", "original", "cuda_cache", "template"):
        if not Path(values[key]).is_absolute() or "," in values[key]:
            raise ValueError(f"{key} must be an absolute path without commas")
    profile = json.loads((HERE / "profile.json").read_text())
    def expand(items):
        return [Template(item).substitute(values) for item in items]
    argv = ["docker", "run", "-d"] + expand(profile["docker_options"])
    for mount in expand(profile["mounts"]):
        argv.extend(["--mount", mount])
    argv.extend(["--entrypoint=/usr/bin/env", values["image"], "-i"])
    argv.extend(expand(profile["environment"]))
    argv.extend(["/usr/local/bin/spark"] + expand(profile["server_argv"]))
    return argv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in PARAMETERS:
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = vars(parser.parse_args())
    output_format = args.pop("format")
    try:
        argv = render(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(argv, indent=2) if output_format == "json" else shlex.join(argv))
