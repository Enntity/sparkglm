#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Translate the managed recipe contract into the qualified Atlas launch profile."""
import ipaddress
import json
import os
from pathlib import Path
import re


def fabric_hcas(value, sysfs=Path('/sys/class/infiniband')):
    """RDMA devices for the fabric: the requested ones plus, for each, the
    active device on the same physical port behind the other PCIe domain.

    GB10 attaches its ConnectX-7 through two PCIe x4 links, so one 200G cable
    shows up as two RDMA devices (e.g. rocep1s0f0 and roceP2p1s0f0), each
    capped near 112 Gb/s. Filling the cable needs both.
    """
    names = value.split(',')
    if not all(re.fullmatch(r'[A-Za-z0-9_.:-]+', name) for name in names):
        raise ValueError('Invalid fabric interface or HCA')

    def pci(dev):
        try:
            return (sysfs/dev/'device').resolve().name
        except OSError:
            return None

    def active(dev):
        try:
            return 'ACTIVE' in (sysfs/dev/'ports'/'1'/'state').read_text()
        except OSError:
            return False

    hcas = list(names)
    others = sorted(p.name for p in sysfs.iterdir()) if sysfs.is_dir() else []
    for name in names:
        addr = pci(name)
        if not addr or ':' not in addr:
            continue
        slot = addr.split(':', 1)[1]
        for dev in others:
            sibling = pci(dev)
            if (dev not in hcas and sibling and ':' in sibling and sibling != addr
                    and sibling.split(':', 1)[1] == slot and active(dev)):
                hcas.append(dev)
    return hcas


def disk_prefix_cache(gb):
    """Engine settings that keep evicted prefix-cache blocks and recurrent-state
    snapshots on the node's disk, in the directory start-node.sh mounts at
    /prefix-cache. The size (GiB per node) is split evenly between the two;
    empty or 0 leaves the tier off."""
    if gb in ('', '0'):
        return {}
    if not re.fullmatch(r'[1-9][0-9]{1,2}', gb) or not 16 <= int(gb) <= 100:
        raise ValueError('SPARKGLM_PREFIX_CACHE_GB must be a whole number from 16 to 100')
    kv = int(gb) // 2
    return {'ATLAS_KV_NVME_DIR': '/prefix-cache/kv', 'ATLAS_KV_NVME_GB': str(kv),
            'ATLAS_GLM_NVME_FAST': '1', 'ATLAS_SSM_TIER': '1', 'ATLAS_SSM_TIER_UNIFIED': '1',
            'ATLAS_SSM_TIER_SWAP_DIR': '/prefix-cache/ssm', 'ATLAS_SSM_TIER_DISK_GB': str(int(gb) - kv),
            'ATLAS_SSM_TIER_SLOTS': '2'}


def launch(environ, profile):
    rank = environ.get('NODE_RANK', '')
    if rank not in ('0', '1'):
        raise ValueError('NODE_RANK must be 0 or 1')
    master = str(ipaddress.ip_address(environ['MASTER_ADDR']))
    port = int(environ.get('MASTER_PORT', '29510'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid MASTER_PORT')
    interface = environ['FABRIC_INTERFACE']
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', interface):
        raise ValueError('Invalid fabric interface or HCA')
    hcas = fabric_hcas(environ.get('FABRIC_HCA', 'rocep1s0f0'))
    model = environ.get('MODEL_PATH', '/models/atlas-overlay')
    drafter = environ.get('DRAFTER_PATH', '/models/drafter')
    for label, path in (('MODEL_PATH', model), ('DRAFTER_PATH', drafter)):
        if not Path(path).is_absolute() or any(c in path for c in '\r\n\0'):
            raise ValueError(f'{label} must be an absolute path')
    # Profile values intentionally win over ambient optimization flags. Runtime
    # changes require a new reviewed image/profile instead of accidental tuning.
    env = {key: value for key, value in environ.items() if not key.startswith('ATLAS_')}
    env.update(profile['environment'])
    env['NCCL_SOCKET_IFNAME'] = interface
    env['NCCL_IB_HCA'] = ','.join(hcas)
    # The engine's direct RDMA all-reduce stripes over the same devices.
    env['ATLAS_RDMA_RAILS'] = env['NCCL_IB_HCA']
    args = [str(arg).replace('${model}', model).replace('${drafter}', drafter)
            for arg in profile['server_argv']]
    # A dedicated pair can give the KV pool (and so the prefix cache) more of
    # the unified memory than the shipped default leaves free.
    util = environ.get('SPARKGLM_GPU_MEMORY_UTILIZATION')
    if util:
        if not re.fullmatch(r'0\.(8[0-9]|9[0-5])', util):
            raise ValueError('SPARKGLM_GPU_MEMORY_UTILIZATION must be 0.80 to 0.95')
        args = [f'--gpu-memory-utilization={util}' if a.startswith('--gpu-memory-utilization=') else a
                for a in args]
    env.update(disk_prefix_cache(environ.get('SPARKGLM_PREFIX_CACHE_GB', '')))
    name = environ.get('SERVED_MODEL_NAME', 'glm-5.3-flash-atlas')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,127}', name):
        raise ValueError('Invalid SERVED_MODEL_NAME')
    args += [f'--rank={rank}', f'--master-addr={master}', f'--master-port={port}',
             '--bind=127.0.0.1', f'--port={8893 + int(rank)}', f'--model-name={name}']
    return display_carveout(environ, ['/usr/local/bin/spark', *args]), env


# Bind-mounted from the host by start-node.sh and the LLooM recipe, so every
# container on a Spark shares the one lock.
DISPLAY_CARVEOUT_LOCK = '/run/lock/sparkglm/display-carveout.lock'


def display_carveout(environ, argv):
    """`argv` under `spark display-carveout`, which lends the GB10's 2 GiB
    display carveout to the KV cache, when SPARKGLM_DISPLAY_CARVEOUT=1 (default
    off). The container then needs CAP_SYS_ADMIN for the export, which the
    launcher drops before the server starts, and the host's /run/lock/sparkglm;
    it serves without the carveout if it cannot lend it."""
    switch = environ.get('SPARKGLM_DISPLAY_CARVEOUT', '0')
    if switch not in ('0', '1'):
        raise ValueError('SPARKGLM_DISPLAY_CARVEOUT must be 0 or 1')
    if switch == '0':
        return argv
    return [argv[0], 'display-carveout', f'--lock={DISPLAY_CARVEOUT_LOCK}', '--', *argv]


def profile_path(environ, here):
    """The launch profile named by SPARKGLM_PROFILE (default 4x512k)."""
    name = environ.get('SPARKGLM_PROFILE', '4x512k')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
        raise ValueError('Invalid SPARKGLM_PROFILE')
    return here/'profiles'/f'{name}.json'


def main():
    here = Path(__file__).resolve().parent
    profile = profile_path(os.environ, here)
    argv, env = launch(os.environ, json.loads(profile.read_text()))
    model = Path(env.get('MODEL_PATH', '/models/atlas-overlay'))
    receipt = json.loads((model/'conversion.complete.json').read_text())
    if receipt.get('converted_matrices') != 864:
        raise ValueError('The NVIDIA MTP overlay has not completed conversion')
    # Full tensor verification is a separate installation step, before serving.
    if not (model/'config.json').is_file() or not (model/'model.safetensors.index.json').is_file():
        raise ValueError('Incomplete model overlay')
    drafter = Path(env.get('DRAFTER_PATH', '/models/drafter'))
    if not (drafter/'config.json').is_file() or not (drafter/'model.safetensors').is_file():
        raise ValueError('Incomplete DFlash drafter')
    source = json.loads((here/'source-manifest.json').read_text())
    print(json.dumps({'event': 'atlas-recipe-start', 'rank': env['NODE_RANK'], 'profile': profile.stem,
                      'source': f"{source['repository']}@{source['commit']}"}), flush=True)
    os.execve(argv[0], argv, env)


if __name__ == '__main__':
    main()
