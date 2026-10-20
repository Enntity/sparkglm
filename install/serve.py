#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Translate the managed recipe contract into the qualified Atlas launch profile."""
import ipaddress
import json
import os
from pathlib import Path
import re


def fabric_hcas(value, sysfs=Path('/sys/class/infiniband'), siblings=True):
    """RDMA devices for the fabric: the requested ones plus, for each, the
    active device on the same physical port behind the other PCIe domain.

    GB10 attaches its ConnectX-7 through two PCIe x4 links, so one 200G cable
    shows up as two RDMA devices (e.g. rocep1s0f0 and roceP2p1s0f0), each
    capped near 112 Gb/s. Filling the cable needs both.

    Some OEM GB10 boards (ASUS GX10) wire the other domain to a separate
    physical port instead (Enntity/sparkglm#35). A sibling therefore also needs
    a RoCE v2 IPv4 GID, which the engine requires on every rail; an unaddressed
    port is left out. siblings=False (FABRIC_SIBLINGS=0) uses exactly the
    listed devices, for a second port that is addressed but not cabled to the
    other Spark.
    """
    names = value.split(',')
    if (not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) for name in names)
            or len(names) != len(set(names))):
        raise ValueError('FABRIC_HCA must list unique exact RDMA device names, without spaces or :port suffixes')

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

    def roce_v2_ipv4(dev):
        port = sysfs/dev/'ports'/'1'
        try:
            types = list((port/'gid_attrs'/'types').iterdir())
        except OSError:
            return False
        for kind in types:
            try:  # unpopulated GID slots fail to read
                if (kind.read_text().strip() == 'RoCE v2'
                        and (port/'gids'/kind.name).read_text().startswith('0000:0000:0000:0000:0000:ffff:')):
                    return True
            except OSError:
                continue
        return False

    hcas = list(names)
    if not siblings:
        return hcas
    others = sorted(p.name for p in sysfs.iterdir()) if sysfs.is_dir() else []
    for name in names:
        addr = pci(name)
        if not addr or ':' not in addr:
            continue
        slot = addr.split(':', 1)[1]
        for dev in others:
            sibling = pci(dev)
            if (dev not in hcas and sibling and ':' in sibling and sibling != addr
                    and sibling.split(':', 1)[1] == slot and active(dev) and roce_v2_ipv4(dev)):
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


def world_size(profile):
    """The number of Sparks a profile serves on (its --world-size)."""
    sizes = [str(a).split('=', 1)[1] for a in profile['server_argv'] if str(a).startswith('--world-size=')]
    if len(sizes) != 1 or sizes[0] not in ('2', '4'):
        raise ValueError('A profile must serve on 2 or 4 Sparks')
    return int(sizes[0])


def fabric_rails(environ, rank, world, siblings):
    """RDMA devices per link. Two Sparks share one cable, which NCCL and the
    engine's all-reduce both use (FABRIC_HCA). Four Sparks in a daisy chain
    link each rank only to its neighbours: FABRIC_LEFT_HCA toward rank - 1 and
    FABRIC_RIGHT_HCA toward rank + 1. NCCL cannot reach a Spark that is not
    cabled to it, so the engine carries that traffic itself."""
    chain = ('FABRIC_LEFT_HCA', 'FABRIC_RIGHT_HCA')
    if world == 2:
        if any(environ.get(key) for key in chain):
            raise ValueError('FABRIC_LEFT_HCA and FABRIC_RIGHT_HCA are only for a four-Spark chain')
        hcas = ','.join(fabric_hcas(environ.get('FABRIC_HCA', ''), siblings=siblings))
        return {'NCCL_IB_HCA': hcas, 'ATLAS_RDMA_RAILS': hcas}
    if environ.get('FABRIC_HCA'):
        raise ValueError('A four-Spark chain takes FABRIC_LEFT_HCA and FABRIC_RIGHT_HCA, not FABRIC_HCA')
    rails = {}
    for key, side, linked in zip(chain, ('LEFT', 'RIGHT'), (rank > 0, rank < world - 1)):
        value = environ.get(key, '')
        if bool(value) != linked:
            raise ValueError(f'Rank {rank} of {world} {"needs" if linked else "has no neighbour for"} {key}')
        if value:
            rails[f'ATLAS_RDMA_{side}_RAILS'] = ','.join(fabric_hcas(value, siblings=siblings))
    return rails


def kv_shard(environ):
    """Engine settings that split GLM's MLA latent KV between the two ranks,
    when SPARKGLM_KV_SHARD=1: each rank stores the latents of half the blocks,
    so the shared pool holds about 1.8x the tokens for the same memory. The
    disk prefix-cache tier runs beside it: each rank spills and restores only
    the latents it owns."""
    switch = environ.get('SPARKGLM_KV_SHARD', '0')
    if switch not in ('0', '1'):
        raise ValueError('SPARKGLM_KV_SHARD must be 0 or 1')
    if switch == '0':
        return {}
    return {'ATLAS_GLM_KV_SHARD': '1', 'ATLAS_GLM_KV_SHARD_COMPACT': '1'}


def serving_mode(environ, profile, world):
    """Resolve TP2 precision independently of the capacity profile."""
    if world != 2:
        if 'SPARKGLM_MODE' in environ:
            raise ValueError('strict/spark modes are qualified for TP2 only')
        return None, {}
    mode = environ.get('SPARKGLM_MODE', 'strict')
    if mode not in ('strict', 'spark'):
        raise ValueError('SPARKGLM_MODE must be strict or spark')
    required = {
        'ATLAS_GLM_FP4L_NATIVE': '1',
        'ATLAS_GLM_FP4L_MLA_FP8': '1',
        'ATLAS_GLM_FP4L_SHARED_FP8': '1' if mode == 'strict' else '0',
        'ATLAS_GLM_FP4L_DENSE_MXFP8': '1' if mode == 'strict' else '1',
        'ATLAS_GLM_FP4L_KDA_W8A8_PREFILL': '1',
        'ATLAS_GLM_FP4L_INPUT_SCALES': '1',
        'ATLAS_GLM_FP4L_FP8_PREFILL': 'w8a8',
        'ATLAS_GLM_LM_HEAD_MXFP8': '1',
    }
    if mode == 'spark':
        for flag in ('ATLAS_GLM_FP4L_SHARED_FP8',):
            value = profile['environment'].get(flag, required[flag])
            if value not in ('0', '1'):
                raise ValueError(f'{flag} must be 0 or 1')
            required[flag] = value
    for flag, value in required.items():
        if flag in profile['environment'] and profile['environment'][flag] != value:
            raise ValueError(f'{mode} mode requires {flag}={value}')
    flags = dict(required, ATLAS_GLM_SERVING_MODE=mode,
                 ATLAS_GLM_FP4L_KDA_MXFP8_DECODE='1' if mode == 'strict' else '0',
                 ATLAS_GLM_FP4L_DECODE_A16='1' if mode == 'strict' else '0',
                 ATLAS_GLM_FP4L_KDA_TILED_TWIN='0')
    if mode == 'strict':
        forbidden = ('--ngram-speculative', '--self-speculative')
        if any(arg.split('=', 1)[0] in forbidden for arg in profile['server_argv']):
            raise ValueError('strict mode requires serial target decoding')
        # Keep the canonical one-row target path. The engine disables speculative
        # dispatch for every request in strict mode; the unused drafter stays BF16.
        flags.update(ATLAS_NO_DFLASH_DRAFTER_NVFP4='1', ATLAS_HC_PERSEQ_DECODE='1',
                     ATLAS_GLM_CANONICAL_VERIFY='1', ATLAS_GLM_DFLASH='1')
        for flag in (
            'ATLAS_DFLASH_DRAFTER_FP8', 'ATLAS_DFLASH_NVFP4_TC',
            'ATLAS_DFLASH_NVFP4_HEAD', 'ATLAS_DFLASH_MXFP8',
            'ATLAS_DFLASH_MXFP8_HEAD', 'ATLAS_DFLASH_DROP_BF16',
            'ATLAS_DFLASH_CTX_NVFP4', 'ATLAS_DFLASH_CONV_TWIN',
            'ATLAS_GLM_DRAFT_TP', 'ATLAS_GLM_DRAFT_TP_BATCH',
            'ATLAS_GLM_DRAFT_TP_CTX', 'ATLAS_GLM_DRAFT_TP_TOPK',
        ):
            flags[flag] = '0'
    elif 'ATLAS_NO_DFLASH_DRAFTER_NVFP4' in profile['environment']:
        raise ValueError('spark mode requires its reviewed NVFP4 drafter policy')
    return mode, flags


def launch(environ, profile):
    world = world_size(profile)
    mode, precision = serving_mode(environ, profile, world)
    native_mtp = environ.get('SPARKGLM_STRICT_MTP', '0')
    if native_mtp not in ('0', '1') or (native_mtp == '1' and mode != 'strict'):
        raise ValueError('SPARKGLM_STRICT_MTP must be 0 or 1 and requires strict TP2 mode')
    rank = environ.get('NODE_RANK', '')
    if rank not in [str(r) for r in range(world)]:
        raise ValueError(f'NODE_RANK must be 0 to {world - 1}')
    master = str(ipaddress.ip_address(environ['MASTER_ADDR']))
    port = int(environ.get('MASTER_PORT', '29510'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid MASTER_PORT')
    interface = environ['FABRIC_INTERFACE']
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', interface):
        raise ValueError('Invalid fabric interface or HCA')
    siblings = environ.get('FABRIC_SIBLINGS', '0')
    if siblings not in ('0', '1'):
        raise ValueError('FABRIC_SIBLINGS must be 0 or 1')
    rails = fabric_rails(environ, int(rank), world, siblings == '1')
    model = environ.get('MODEL_PATH', '/models/atlas-overlay')
    drafter = environ.get('DRAFTER_PATH', '/models/drafter')
    for label, path in (('MODEL_PATH', model), ('DRAFTER_PATH', drafter)):
        if not Path(path).is_absolute() or any(c in path for c in '\r\n\0'):
            raise ValueError(f'{label} must be an absolute path')
    # Profile values intentionally win over ambient optimization flags. Runtime
    # changes require a new reviewed image/profile instead of accidental tuning.
    env = {key: value for key, value in environ.items() if not key.startswith('ATLAS_')}
    env.update(profile['environment'])
    env.update(precision)
    env['NCCL_SOCKET_IFNAME'] = interface
    env.update(rails)
    args = [str(arg).replace('${model}', model).replace('${drafter}', drafter)
            for arg in profile['server_argv']]
    if native_mtp == '1':
        # Native prediction retains the selected serving capacity and prefix
        # cache. disable_mtp compares serial on this same loaded target.
        drop = ('--dflash', '--draft-model', '--dflash-gamma', '--dflash-window-size',
                '--num-drafts', '--mtp-quantization', '--mtp-vocab')
        args = [a for a in args if a.split('=', 1)[0] not in drop]
        args += ['--speculative', '--num-drafts=6', '--mtp-quantization=bf16', '--mtp-vocab=0']
        env.update(ATLAS_GLM_STRICT_NATIVE_MTP='1', ATLAS_GLM_DFLASH='0',
                   ATLAS_GLM_MTP_LONG_CONTEXT='1', ATLAS_GLM_NATIVE_MTP_HEAD='bf16',
                   ATLAS_GLM_MTP_BF16_DRAFTS='1', ATLAS_GLM_NATIVE_MTP_REPAIR_BATCH='0',
                   ATLAS_GLM_NATIVE_MTP_STOP_PROB='0.95', ATLAS_MTP_SPEC_THINK='1',
                   ATLAS_NO_MTP_K_LADDER='1', ATLAS_MTP_TIMING='1',
                   # The grouped multi-owner FFN changes target arithmetic.
                   # Keep C4 admission but verify owners individually in strict.
                   ATLAS_GLM_LONG_BATCH_VERIFY='0')
    # A dedicated pair can give the KV pool (and so the prefix cache) more of
    # the unified memory than the shipped default leaves free. Four Sparks hold
    # a quarter of the weights each, so a chain may also leave the host more.
    util = environ.get('SPARKGLM_GPU_MEMORY_UTILIZATION')
    if util:
        low = '6' if world == 4 else '8'
        if not re.fullmatch(rf'0\.([{low}-8][0-9]|9[0-5])', util):
            raise ValueError(f'SPARKGLM_GPU_MEMORY_UTILIZATION must be 0.{low}0 to 0.95')
        args = [f'--gpu-memory-utilization={util}' if a.startswith('--gpu-memory-utilization=') else a
                for a in args]
    env.update(disk_prefix_cache(environ.get('SPARKGLM_PREFIX_CACHE_GB', '')))
    if mode:
        env['SPARKGLM_MODE'] = mode
        for flag in ('ATLAS_KV_NVME_DIR', 'ATLAS_SSM_TIER_SWAP_DIR'):
            if flag in env:
                env[flag] = env[flag].replace('/prefix-cache/', f'/prefix-cache/{mode}/', 1)
    env.update(kv_shard(environ))
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
    off). The container needs the host's /run/lock/sparkglm; it serves without
    the carveout if the driver cannot lend it."""
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


def mtp_ready(model):
    """Whether the checkpoint at `model` carries the quantized MTP layer the
    engine serves. FP4-Local ships its MTP experts quantized (W4A16 NVFP4)
    and is served as published; NVIDIA's earlier NVFP4 release ships them in
    BF16 and is served through the overlay install/convert.sh writes, whose
    receipt counts the 864 matrices it quantized."""
    receipt = model/'conversion.complete.json'
    if receipt.is_file():
        return json.loads(receipt.read_text()).get('converted_matrices') == 864
    config = json.loads((model/'config.json').read_text())
    mtp = (config.get('text_config') or config).get('num_hidden_layers')
    quantized = (config.get('quantization_config') or {}).get('quantized_layers') or {}
    return f'model.language_model.layers.{mtp}.mlp.experts' in quantized


def main():
    here = Path(__file__).resolve().parent
    profile = profile_path(os.environ, here)
    argv, env = launch(os.environ, json.loads(profile.read_text()))
    model = Path(env.get('MODEL_PATH', '/models/atlas-overlay'))
    if not (model/'config.json').is_file() or not (model/'model.safetensors.index.json').is_file():
        raise ValueError('Incomplete checkpoint')
    # Full tensor verification of an overlay is a separate installation step, before serving.
    if not mtp_ready(model):
        raise ValueError('The NVIDIA NVFP4 checkpoint needs the MTP overlay (install/convert.sh), '
                         'which has not completed conversion')
    drafter = Path(env.get('DRAFTER_PATH', '/models/drafter'))
    if env.get('ATLAS_GLM_STRICT_NATIVE_MTP') != '1' and (not (drafter/'config.json').is_file() or not (drafter/'model.safetensors').is_file()):
        raise ValueError('Incomplete DFlash drafter')
    source = json.loads((here/'source-manifest.json').read_text())
    print(json.dumps({'event': 'atlas-recipe-start', 'rank': env['NODE_RANK'], 'profile': profile.stem,
                      'mode': env.get('SPARKGLM_MODE', 'legacy-tp4'),
                      'target_kda_decode': 'nvfp4-rne' if env.get('SPARKGLM_MODE') == 'spark' else 'native',
                      'source': f"{source['repository']}@{source['commit']}"}), flush=True)
    os.execve(argv[0], argv, env)


if __name__ == '__main__':
    main()
