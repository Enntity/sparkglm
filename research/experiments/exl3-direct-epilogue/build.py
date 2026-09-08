#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print or build a declared candidate; do not start/stop any server."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from check_video_source_parity import candidate_expectations
from patch import transform, REFERENCE_SHA256


def declaration():
    candidate = transform((ROOT / 'overlay/exl3_fat_gemm.cu').read_text())
    return {'schema': 'sparkglm.candidate-sources/v1', 'changes': {'exl3': {
        'exl3-fat-kernel/exl3_fat_gemm.cu': {
            'reference_sha256': REFERENCE_SHA256,
            'candidate_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
            'reason': 'Experimental warp-owned direct Hadamard epilogue for grouped prefill only; no GPU qualification yet.',
        }
    }}}


def render():
    path = HERE / 'sources.json'
    manifest = json.loads(path.read_text())
    if manifest != declaration():
        raise ValueError('declared candidate differs from generated source; review sources.json')
    frozen = json.loads((ROOT / 'provenance/video-source-parity.json').read_text())
    candidate_expectations(frozen, manifest, 'exl3', {}, {})
    # Only EXL3 changes. Keep the complete vLLM build prefix byte-identical,
    # including its reference gate, so this does not invalidate native caches.
    text = (ROOT / 'Dockerfile').read_text()
    gate = 'RUN python3 /opt/glm53/check_video_source_parity.py --kind exl3'
    if text.count(gate) != 1:
        raise ValueError('reference EXL3 gate layout changed')
    text = text.replace(gate,
        f'COPY {path.relative_to(ROOT).as_posix()} /opt/sparkglm-candidate-sources.json\n' +
        'RUN python3 /opt/glm53/check_video_source_parity.py --candidate-manifest /opt/sparkglm-candidate-sources.json --kind exl3')
    anchor = 'COPY overlay/exl3_fat_gemm.cu /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu\n'
    if text.count(anchor) != 1:
        raise ValueError('reference COPY layout changed')
    text = text.replace(anchor, anchor +
        'COPY research/experiments/exl3-direct-epilogue/patch.py /opt/glm53/direct-epilogue.py\n'
        'RUN python3 /opt/glm53/direct-epilogue.py --apply /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return text + (f'\nLABEL org.enntity.sparkglm.build-profile="candidate" '
                   f'org.enntity.sparkglm.candidate-manifest="sha256:{digest}"\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-dockerfile', action='store_true')
    parser.add_argument('--tag', default='sparkglm-candidate:direct-epilogue')
    args = parser.parse_args()
    if not re.fullmatch(r'sparkglm-candidate:[A-Za-z0-9_][A-Za-z0-9_.-]*', args.tag):
        parser.error('only local sparkglm-candidate tags are permitted')
    text = render()
    if args.print_dockerfile:
        print(text, end='')
        return
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        parser.error('commit candidate changes before building')
    meminfo = Path('/proc/meminfo')
    available = re.search(r'^MemAvailable:\s+(\d+)', meminfo.read_text(), re.M) if meminfo.exists() else None
    if available is None or int(available[1]) < 32 * 1024 * 1024:
        parser.error('build on a reserved Spark with at least 32 GiB MemAvailable')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix='sparkglm-direct-epilogue-') as temporary:
        dockerfile = Path(temporary) / 'Dockerfile'
        dockerfile.write_text(text)
        subprocess.run(['docker', 'build', '--progress=plain', '-f', str(dockerfile),
                        '--build-arg', f'SPARKGLM_SOURCE_REVISION={revision}',
                        '-t', args.tag, str(ROOT)], check=True)


if __name__ == '__main__':
    main()
