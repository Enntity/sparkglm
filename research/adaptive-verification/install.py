#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Apply the retained upstream patch only to the exact measured base sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent

def digest(data):
    return hashlib.sha256(data).hexdigest()

def install(root):
    contract = json.loads((HERE / 'source-contract.json').read_text())
    patcher = HERE / 'vendor/mia/patch_adaptive_k.py'
    if digest(patcher.read_bytes()) != contract['patch_sha256']:
        raise ValueError('adaptive patch source differs from the measured pin')
    entries = contract['files']
    sources = [(root / e['path']).read_bytes() for e in entries]
    hashes = [digest(data) for data in sources]
    if hashes == [e['result_sha256'] for e in entries]:
        return
    if hashes != [e['base_sha256'] for e in entries]:
        raise ValueError('unqualified or partially patched vLLM source; no files changed')
    with tempfile.TemporaryDirectory() as temp:
        staged = [Path(temp) / Path(e['path']).name for e in entries]
        for path, data in zip(staged, sources):
            path.write_bytes(data)
        env = dict(os.environ, GLM53_SCHEDULER_PY=str(staged[0]),
                   GLM53_CUDAGRAPH_UTILS_PY=str(staged[1]))
        subprocess.run([sys.executable, '-S', str(patcher)], env=env, check=True)
        results = [path.read_bytes() for path in staged]
        for e, data in zip(entries, results):
            if digest(data) != e['result_sha256']:
                raise ValueError('patch output differs from measured source: ' + e['path'])
            compile(data, e['path'], 'exec')
        # All checks finish before either installed file changes. A failed Docker
        # RUN never commits a partially written layer; do not run in live servers.
        for e, data in zip(entries, results):
            (root / e['path']).write_bytes(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/usr/local/lib/python3.12/dist-packages/vllm'))
    install(parser.parse_args().root)
    print('Adaptive source composition matches the measured hashes.')
