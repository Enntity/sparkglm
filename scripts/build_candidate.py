#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build an explicitly labeled experiment without changing the reference Dockerfile."""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from check_video_source_parity import candidate_expectations

ROOT = Path(__file__).resolve().parents[1]


def render(dockerfile: str, relative_manifest: str, digest: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", relative_manifest):
        raise ValueError("manifest path must contain only letters, digits, _, ., /, -")
    for destination in ("/opt/video-source-parity.json", "/opt/glm53/video-source-parity.json"):
        anchor = f"COPY provenance/video-source-parity.json {destination}\n"
        if dockerfile.count(anchor) != 1:
            raise ValueError("reference Dockerfile layout changed; review candidate integration")
        dockerfile = dockerfile.replace(anchor, anchor +
            f"COPY {relative_manifest} /opt/sparkglm-candidate-sources.json\n")
    for kind in ("native", "python", "exl3"):
        anchor = f"check_video_source_parity.py --kind {kind}"
        if dockerfile.count(anchor) != 1:
            raise ValueError(f"missing or ambiguous {kind} source gate")
        dockerfile = dockerfile.replace(anchor,
            f"check_video_source_parity.py --candidate-manifest /opt/sparkglm-candidate-sources.json --kind {kind}")
    return dockerfile + (f'\nLABEL org.enntity.sparkglm.build-profile="candidate" \\\n'
                         f'      org.enntity.sparkglm.candidate-manifest="sha256:{digest}"\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="local sparkglm-candidate:NAME tag")
    parser.add_argument("--print-dockerfile", action="store_true", help="inspect only; no Docker/GPU access")
    args = parser.parse_args()
    if not re.fullmatch(r"sparkglm-candidate:[A-Za-z0-9_][A-Za-z0-9_.-]*", args.tag):
        parser.error("use a local sparkglm-candidate:NAME tag, never the serving image")
    try:
        relative = args.manifest.resolve().relative_to(ROOT).as_posix()
        raw = args.manifest.read_bytes()
        declaration = json.loads(raw)
        frozen = json.loads((ROOT / "provenance/video-source-parity.json").read_text())
        candidate_expectations(frozen, declaration, "python", {}, {})
        recipe = render((ROOT / "Dockerfile").read_text(), relative, hashlib.sha256(raw).hexdigest())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.print_dockerfile:
        print(recipe, end="")
        return
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        parser.error("commit the candidate source and declaration before building; working tree is dirty")
    # The builder does not source .env or touch a running model/container.
    memory = Path("/proc/meminfo")
    available = re.search(r"^MemAvailable:\s+(\d+)", memory.read_text(), re.M) if memory.exists() else None
    if not available or int(available[1]) < 32 * 1024 * 1024:
        parser.error("build on the Spark with at least 32 GiB MemAvailable; stop resident models first")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="sparkglm-candidate-") as temporary:
        path = Path(temporary) / "Dockerfile"
        path.write_text(recipe)
        subprocess.run(["docker", "build", "--progress=plain", "-f", str(path),
                        "--build-arg", f"SPARKGLM_SOURCE_REVISION={revision}",
                        "-t", args.tag, str(ROOT)], check=True)
    print(f"Built {args.tag}: unqualified candidate, not the default or video reference.")


if __name__ == "__main__":
    main()
