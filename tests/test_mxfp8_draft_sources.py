#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep the reviewed draft compatibility code tied to its exact provenance."""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parents[1] / 'research/experiments/mxfp8-draft'
pin = json.loads((root / 'upstream.json').read_text())
assert pin['upstream']['revision'] == 'd00b2ffa70e0ccbfd582088ab32448b01bfe9c67'
assert pin['model']['revision'] == '610aa967a92bfeb97e3d848dcb8693553e8b6a55'
for entry in pin['files']:
    source = root / entry['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256'], source
    if source.suffix == '.py':
        compile(source.read_text(), str(source), 'exec')
print('Pinned MXFP8 draft sources: PASS')
