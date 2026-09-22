# SPDX-License-Identifier: Apache-2.0
"""Install the pinned hybrid offloading changes without replacing whole files.

Hybrid-group changes adapted from Mia recipe PR232 at
3c2add4c491737c5b916217e0c5d6dc708fe02d8. Transfer-failure propagation and
all-rank-fenced cold retry are original SparkGLM integration work.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(text, entry):
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest == entry['result_sha256']:
        return text
    if digest != entry['base_sha256']:
        raise ValueError('unqualified source revision: ' + entry['path'])
    for edit in entry['edits']:
        if text.count(edit['before']) != 1:
            raise ValueError('ambiguous patch anchor: ' + entry['path'])
        text = text.replace(edit['before'], edit['after'], 1)
    if hashlib.sha256(text.encode()).hexdigest() != entry['result_sha256']:
        raise ValueError('patch output mismatch: ' + entry['path'])
    compile(text, entry['path'], 'exec')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/usr/local/lib/python3.12/dist-packages/vllm')
    parser.add_argument('--patch', default=str(Path(__file__).parent / 'kvoffload/hybrid_patch.json'))
    args = parser.parse_args()
    entries = json.loads(Path(args.patch).read_text())['files']
    # Preflight every target before writing any file.
    prepared = [(Path(args.root) / e['path'], prepare((Path(args.root) / e['path']).read_text(), e)) for e in entries]
    for path, text in prepared:
        if path.read_text() != text:
            path.write_text(text)
    print('NVMe hybrid offloading patch verified:', len(prepared), 'files')


if __name__ == '__main__':
    main()
