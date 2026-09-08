#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install and serve SparkGLM directly with Docker and SSH; no gateway required."""
from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

from appliance import ROOT, build, materialize, remote, run

OWNER = 'org.enntity.sparkglm.owner=standalone'


def command(config, rank, *args, capture=False):
    if rank:
        return remote(config['worker'], *args, capture=capture)
    return run(*args, capture=capture)


def containers(config, action):
    """Operate only containers bearing our ownership label, never LLooM's."""
    for rank in (0, 1):
        name = config['names'][rank]
        ids = command(config, rank, 'docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$', capture=True)
        if not ids:
            continue
        owned = command(config, rank, 'docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$', '--filter', 'label=' + OWNER, capture=True)
        if ids != owned:
            raise RuntimeError('Refusing to operate on a container not owned by this standalone installer: ' + name)
        if action == 'stop':
            command(config, rank, 'docker', 'stop', '--time', '30', name)
        elif action == 'remove':
            command(config, rank, 'docker', 'rm', '-f', name)
        elif action == 'status':
            print(command(config, rank, 'docker', 'inspect', '--format', '{{.Name}} {{.State.Status}} {{.Image}}', name, capture=True))


def docker_args(recipe, config, rank):
    member = recipe['models'][0]['settings']['placement']['members'][1 if rank == 0 else 0]
    source = member['runtimeSettings']['bootstrap']['createArgs']
    address = config['head_address'] if rank == 0 else config['worker_address']
    substitutions = {'modelRoot':config['model_root'], 'nodeAddress':address,
                     'fabricInterface':config['interfaces'][rank], 'nodeRank':str(rank),
                     'clusterNodeCount':'2', 'leaderAddress':config['head_address']}
    result = ['docker', 'run', '-d', '--name', config['names'][rank], '--label', OWNER]
    for value in source:
        if value.startswith('type=bind,src=${lloomRoot}'):
            value = 'type=bind,src=' + config['entrypoint'] + ',dst=/opt/sparkglm/entrypoint.sh,readonly'
        for key, replacement in substitutions.items():
            value = value.replace('${' + key + '}', replacement)
        if '${' in value:
            raise RuntimeError('Unresolved runtime parameter: ' + value)
        value = value.replace('lloom-glm53-', 'sparkglm-standalone-')
        if value.startswith('VLLM_HOST='):
            value = 'VLLM_HOST=' + address
        result.append(value)
    result += ['-e', 'VLLM_HOST_IP=' + address,
               '-e', 'NCCL_IB_HCA=' + config['hcas'][rank],
               '-e', 'NCCL_IB_GID_INDEX=' + str(config['gids'][rank]),
               config['images'][rank], '/opt/sparkglm/entrypoint.sh']
    return result


def start(config, timeout):
    recipe = materialize(config['images'][0], config['skip'], config['profile'])
    # Only replace this installer's named containers. A failed start cleans up both ranks.
    containers(config, 'remove')
    try:
        command(config, 1, *docker_args(recipe, config, 1))
        command(config, 0, *docker_args(recipe, config, 0))
        port = recipe['models'][0]['settings']['port']
        model = recipe['models'][0]['gatewayModel']
        url = f"http://{config['head_address']}:{port}/v1/models"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for rank in (0, 1):
                running = command(config, rank, 'docker', 'inspect', '--format', '{{.State.Running}}', config['names'][rank], capture=True)
                if running != 'true':
                    raise RuntimeError('Rank exited; inspect docker logs for ' + config['names'][rank])
            try:
                with opener.open(url, timeout=5) as response:
                    models = json.load(response).get('data', [])
                if any(item.get('id') == model for item in models):
                    print(f'Ready: {url.removesuffix("/models")} model={model}')
                    return
            except (urllib.error.URLError, TimeoutError, ValueError):
                pass
            time.sleep(5)
        raise RuntimeError('Model readiness timed out; inspect the rank logs before retrying.')
    except BaseException:
        containers(config, 'stop')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='install', choices=['plan','build','install','start','stop','status'])
    parser.add_argument('--profile', choices=['nvfp4','exl3'], default='nvfp4')
    parser.add_argument('--worker', help='SSH destination USER@WORKER')
    parser.add_argument('--head-address', help='Leader IPv4 address on the private TP fabric')
    parser.add_argument('--worker-address', help='Worker IPv4 address on the private TP fabric')
    parser.add_argument('--interface', help='Leader fabric interface')
    parser.add_argument('--worker-interface', help='Worker interface; defaults to --interface')
    parser.add_argument('--hca', help='Leader RDMA device from ibdev2netdev')
    parser.add_argument('--worker-hca', help='Worker RDMA device; defaults to --hca')
    parser.add_argument('--gid', type=int, help='Leader populated RoCE v2 GID index')
    parser.add_argument('--worker-gid', type=int, help='Worker GID index; defaults to --gid')
    parser.add_argument('--image', help='Reuse a qualified immutable local image ID')
    parser.add_argument('--skip', action='store_true', help='Recorded NVFP4 skip comparison option')
    parser.add_argument('--model-root', type=Path, default=Path.home()/'.cache/sparkglm/models')
    parser.add_argument('--ready-timeout', type=int, default=7200)
    args = parser.parse_args()
    if args.skip and args.profile == 'exl3':
        parser.error('--skip is the recorded NVFP4 option')
    if args.ready_timeout <= 0:
        parser.error('--ready-timeout must be positive')
    if args.command == 'plan':
        recipe = materialize('sha256:'+'0'*64, args.skip, args.profile)
        manifest = 'build-exl3.json' if args.profile == 'exl3' else 'build.json'
        print(json.dumps({'manager':'standalone', 'profile':recipe,
                          'build':json.loads((ROOT/'profiles'/manifest).read_text()),
                          'scheduling':'skip' if args.skip else 'mixed'}, indent=2))
        return
    cache = Path.home()/'.cache/sparkglm'
    cache.mkdir(parents=True, exist_ok=True)
    state = cache/('standalone-'+args.profile+'.json')
    if args.command in ('start','stop','status'):
        if not state.exists():
            parser.error('No standalone installation exists for this profile; run install first.')
        config = json.loads(state.read_text())
        if args.command == 'start':
            start(config, args.ready_timeout)
        else:
            containers(config, args.command)
        return
    if args.command == 'build':
        print(build(cache, args.profile))
        return
    for flag in ('worker','head_address','worker_address','interface','hca','gid'):
        if getattr(args, flag) is None:
            parser.error('--'+flag.replace('_','-')+' is required; see SPARKGLM.md')
    if args.worker.startswith('-'):
        parser.error('Invalid worker SSH destination')
    for address in (args.head_address, args.worker_address):
        ipaddress.IPv4Address(address)
    for value in (args.interface,args.worker_interface or args.interface,args.hca,args.worker_hca or args.hca):
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+',value):
            parser.error('Invalid fabric interface or RDMA device')
    gids = [args.gid, args.worker_gid if args.worker_gid is not None else args.gid]
    if any(x < 0 for x in gids):
        parser.error('GID indices must be nonnegative')
    model_root = str(args.model_root.expanduser().resolve())
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', model_root):
        parser.error('Use a model root without spaces or shell metacharacters')
    # Matching paths simplify the read-only two-rank mounts; no gateway is needed.
    if remote(args.worker, 'printenv', 'HOME', capture=True) != str(Path.home()):
        parser.error('Use matching account home paths on the two Sparks')
    run('docker','version')
    remote(args.worker,'docker','version')
    image = args.image or build(cache,args.profile)
    recipe = materialize(image,args.skip,args.profile)
    if run('docker','image','inspect','--format','{{.Id}}',image,capture=True) != image:
        raise RuntimeError('Local image identity mismatch')
    try:
        worker_image = remote(args.worker,'docker','image','inspect','--format','{{.Id}}',image,capture=True)
    except subprocess.CalledProcessError:
        worker_image = ''
    if worker_image != image:
        save = subprocess.Popen(['docker','save',image],stdout=subprocess.PIPE)
        load = subprocess.run(['ssh','-o','BatchMode=yes',args.worker,'docker load'],stdin=save.stdout)
        save.stdout.close()
        if save.wait() or load.returncode:
            raise RuntimeError('Image transfer failed')
        if remote(args.worker,'docker','image','inspect','--format','{{.Id}}',image,capture=True) != image:
            raise RuntimeError('Worker image identity mismatch')
    env = cache/'hf-cli'
    if not (env/'bin/hf').exists():
        run('python3','-m','venv',str(env))
        run(str(env/'bin/pip'),'install','huggingface_hub[cli]==0.35.3')
    for step in recipe['setup']['steps']:
        if step.get('action') == 'download-model':
            directory = str(Path(model_root)/step['model'].replace('/','--'))
            run(str(env/'bin/hf'),'download',step['model'],'--revision',step['revision'],'--local-dir',directory)
            remote(args.worker,'mkdir','-p',directory)
            run('rsync','-a','--partial',directory+'/',args.worker+':'+directory+'/')
    entrypoint = cache/'runtime-entrypoint.sh'
    entrypoint.write_bytes((ROOT/'runtime/entrypoint.sh').read_bytes())
    remote(args.worker,'mkdir','-p',str(cache))
    run('scp',str(entrypoint),args.worker+':'+str(entrypoint))
    config = dict(profile=args.profile,worker=args.worker,head_address=args.head_address,
                  worker_address=args.worker_address,model_root=model_root,entrypoint=str(entrypoint),
                  interfaces=[args.interface,args.worker_interface or args.interface],
                  hcas=[args.hca,args.worker_hca or args.hca],gids=gids,images=[image,image],skip=args.skip,
                  names=['sparkglm-standalone-'+args.profile+'-head','sparkglm-standalone-'+args.profile+'-worker'])
    state.write_text(json.dumps(config,indent=2)+'\n')
    start(config,args.ready_timeout)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
