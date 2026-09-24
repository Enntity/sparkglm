#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Translate the managed recipe contract into the qualified Atlas launch profile."""
import ipaddress
import json
import os
from pathlib import Path
import re


def launch(environ, profile):
    rank = environ.get('NODE_RANK', '')
    if rank not in ('0', '1'):
        raise ValueError('NODE_RANK must be 0 or 1')
    master = str(ipaddress.ip_address(environ['MASTER_ADDR']))
    port = int(environ.get('MASTER_PORT', '29510'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid MASTER_PORT')
    interface = environ['FABRIC_INTERFACE']
    hca = environ.get('FABRIC_HCA', 'rocep1s0f0')
    for value in (interface, hca):
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', value):
            raise ValueError('Invalid fabric interface or HCA')
    model = environ.get('MODEL_PATH', '/models/atlas-overlay')
    if not Path(model).is_absolute() or any(c in model for c in '\r\n\0'):
        raise ValueError('MODEL_PATH must be an absolute path')
    # Profile values intentionally win over ambient optimization flags. Runtime
    # changes require a new reviewed image/profile instead of accidental tuning.
    env = {key: value for key, value in environ.items() if not key.startswith('ATLAS_')}
    env.update(profile['environment'])
    env['NCCL_SOCKET_IFNAME'] = interface
    env['NCCL_IB_HCA'] = hca
    args = [str(arg).replace('${model}', model) for arg in profile['server_argv']]
    args += [f'--rank={rank}', f'--master-addr={master}', f'--master-port={port}',
             '--bind=127.0.0.1', f'--port={8893 + int(rank)}']
    return ['/usr/local/bin/spark', *args], env


def main():
    here = Path(__file__).resolve().parent
    argv, env = launch(os.environ, json.loads((here/'profile.json').read_text()))
    model = Path(env.get('MODEL_PATH', '/models/atlas-overlay'))
    receipt = json.loads((model/'conversion.complete.json').read_text())
    if receipt.get('converted_matrices') != 864:
        raise ValueError('The NVIDIA MTP overlay has not completed conversion')
    # Full tensor verification is a separate installation step, before serving.
    if not (model/'config.json').is_file() or not (model/'model.safetensors.index.json').is_file():
        raise ValueError('Incomplete model overlay')
    print(json.dumps({'event': 'atlas-recipe-start', 'rank': env['NODE_RANK'],
                      'source': json.loads((here/'source-manifest.json').read_text())['engine_revision']}), flush=True)
    os.execve(argv[0], argv, env)


if __name__ == '__main__':
    main()
