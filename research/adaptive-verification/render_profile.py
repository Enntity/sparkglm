#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Render a portable local-image profile; does not contact or change a server."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

def render(head_image, worker_image):
    for image in (head_image, worker_image):
        if not re.fullmatch(r'(?:[A-Za-z0-9._:/-]+@)?sha256:[0-9a-f]{64}', image):
            raise ValueError('Use immutable sha256 image IDs or digest references')
    profile = json.loads((ROOT / 'profiles/adaptive-dflash2-nvfp4-nvidia.json').read_text())
    identities = {'head': head_image, 'worker': worker_image}
    # Model pins, topology, KV settings and every template setting participate.
    namespace = hashlib.sha256(json.dumps({'profile': profile, 'images': identities}, sort_keys=True).encode()).hexdigest()
    for member in profile['models'][0]['settings']['placement']['members']:
        bootstrap = member['runtimeSettings']['bootstrap']
        bootstrap['image'] = identities[member['role']]
        bootstrap['createArgs'] = [arg.replace('SPARKGLM_NVME_NAMESPACE=RENDER_REQUIRED',
                                              'SPARKGLM_NVME_NAMESPACE=' + namespace)
                                   for arg in bootstrap['createArgs']]
    return profile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--head-image', required=True)
    parser.add_argument('--worker-image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(render(args.head_image, args.worker_image), indent=2) + '\n')
