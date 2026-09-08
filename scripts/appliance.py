#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the measured NVFP4 layers; let LLooM install and own the TP2 runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RECIPE_ID = 'linux-nvidia-dgx-spark-2x-sparkglm-nvfp4'
RUNTIME_ID = 'sparkglm-nvfp4-cluster'


def run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ''


def remote(worker: str, *args: str, capture: bool = False) -> str:
    return run('ssh', '-o', 'BatchMode=yes', worker, shlex.join(args), capture=capture)


def build(cache: Path, profile: str = 'nvfp4') -> str:
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('Build on the Spark leader (Linux ARM64), not a Mac or x86 machine.')
    available = next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if available < 32 * 1024**2:
        raise RuntimeError('Native compilation needs 32 GiB MemAvailable. Stop resident models through their runtime manager before building.')
    manifest = json.loads((ROOT / ('profiles/build-exl3.json' if profile == 'exl3' else 'profiles/build.json')).read_text())
    previous = None
    for layer in manifest['layers']:
        revision = layer['revision']
        try:
            run('git', '-C', str(ROOT), 'cat-file', '-e', revision + '^{commit}', capture=True)
        except subprocess.CalledProcessError:
            run('git', '-C', str(ROOT), 'fetch', 'origin', revision)
        with tempfile.TemporaryDirectory(prefix='source-', dir=cache) as tmp:
            archive = Path(tmp) / 'source.tar'
            run('git', '-C', str(ROOT), 'archive', '-o', str(archive), revision)
            run('tar', '-xf', str(archive), '-C', tmp)
            archive.unlink()
            tag = 'sparkglm-source:' + revision[:12]
            command = ['docker', 'build', '--platform', 'linux/arm64', '-t', tag,
                       '-f', str(Path(tmp) / layer['dockerfile']),
                       '--build-arg', 'SPARKGLM_SOURCE_REVISION=' + revision]
            if previous:
                command += ['--build-arg', 'BASE_IMAGE=' + previous]
            run(*command, tmp)
            previous = tag
    return run('docker', 'image', 'inspect', '--format', '{{.Id}}', previous, capture=True)


def materialize(image: str, skip: bool = False, profile: str = 'nvfp4') -> dict:
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Use a full immutable Docker image ID: sha256: plus 64 hex digits.')
    recipe = json.loads((ROOT / 'profiles' / (profile + '.json')).read_text())
    for member in recipe['models'][0]['settings']['placement']['members']:
        bootstrap = member['runtimeSettings']['bootstrap']
        bootstrap['image'] = image
        if skip:
            bootstrap['createArgs'] = [x.replace('GLM53_MIXED_PREFILL_CHUNK=0', 'GLM53_MIXED_PREFILL_CHUNK=skip') for x in bootstrap['createArgs']]
    return recipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['nvfp4', 'exl3'], default='nvfp4', help='NVFP4 is the default; EXL3 selects the latest concurrent E3 profile')
    parser.add_argument('command', nargs='?', default='install', choices=['plan', 'build', 'check', 'install', 'start', 'stop', 'status'])
    parser.add_argument('--worker', help='SSH destination of the LLooM worker; required for install')
    parser.add_argument('--image', help='Reuse a qualified local immutable image instead of rebuilding')
    parser.add_argument('--worker-image', help='Reuse a separately qualified worker image; requires --image')
    parser.add_argument('--skip', action='store_true', help='Use the recorded skip comparison setting instead of default mixed scheduling')
    parser.add_argument('--model-root', type=Path, default=Path.home() / '.lloom/models')
    parser.add_argument('--lloom-root', type=Path, help='Installed LLooM source root (normally inferred from its executable)')
    args = parser.parse_args()
    profile = args.profile
    selected = json.loads((ROOT / 'profiles' / (profile + '.json')).read_text())
    recipe_id = selected['id']
    runtime_id = selected['models'][0]['runtime']
    if args.skip and profile == 'exl3':
        parser.error('--skip is the recorded NVFP4 comparison option; EXL3 uses its measured mixed profile.')
    if args.command == 'check' and not args.image:
        parser.error('check requires --image; it never builds or installs a runtime.')
    if args.command == 'plan':
        if args.skip:
            for member in selected['models'][0]['settings']['placement']['members']:
                boot = member['runtimeSettings']['bootstrap']
                boot['createArgs'] = [x.replace('GLM53_MIXED_PREFILL_CHUNK=0', 'GLM53_MIXED_PREFILL_CHUNK=skip') for x in boot['createArgs']]
        print(json.dumps({'profile': selected,
                          'build': json.loads((ROOT / ('profiles/build-exl3.json' if profile == 'exl3' else 'profiles/build.json')).read_text()),
                          'scheduling': 'skip' if args.skip else 'mixed'}, indent=2))
        return
    if args.command in ('start', 'stop', 'status'):
        if args.image or args.worker_image or args.skip:
            parser.error('Apply changed image/scheduling with install first; lifecycle commands use the installed recipe.')
        run('lloom', 'runtime-' + args.command, runtime_id, '--json')
        return
    cache = Path.home() / '.cache/sparkglm'
    cache.mkdir(parents=True, exist_ok=True)
    if args.command == 'build':
        print(build(cache, profile))
        return
    if not args.worker or args.worker.startswith('-'):
        parser.error('Run on the Spark leader with --worker USER@WORKER. See SPARKGLM.md; use plan for a read-only view.')
    executable = shutil.which('lloom')
    if not executable:
        raise RuntimeError('Install and configure LLooM on both Sparks first; see SPARKGLM.md.')
    lloom_root = args.lloom_root or Path(executable).resolve().parents[1]
    expected = json.loads((ROOT / ('profiles/build-exl3.json' if profile == 'exl3' else 'profiles/build.json')).read_text())['lloom_entrypoint_sha256']
    entry = lloom_root / 'backends/sparkglm/entrypoint.sh'
    if hashlib.sha256(entry.read_bytes()).hexdigest() != expected:
        raise RuntimeError('LLooM SparkGLM entrypoint differs from the measured version. Install the pinned LLooM revision in profiles/build.json.')
    # Distributed setup uses the same paths and installed adapter on both nodes.
    worker_home = remote(args.worker, 'printenv', 'HOME', capture=True)
    if worker_home != str(Path.home()):
        raise RuntimeError('This install requires the same home path on both Sparks; configure matching accounts first.')
    worker_hash = remote(args.worker, 'sha256sum', str(entry), capture=True).split()[0]
    if worker_hash != expected:
        raise RuntimeError('Worker LLooM entrypoint differs; install the same LLooM revision and path on both nodes.')
    model_root = args.model_root.expanduser().resolve()
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(model_root)):
        raise RuntimeError('Use a model-root path without spaces or shell metacharacters.')
    image = args.image or build(cache, profile)
    recipe = materialize(image, args.skip, profile)
    worker_expected = args.worker_image or image
    if args.worker_image:
        if not args.image:
            parser.error('--worker-image requires --image')
        materialize(worker_expected)  # validate before sending it to SSH/Docker
        recipe['models'][0]['settings']['placement']['members'][0]['runtimeSettings']['bootstrap']['image'] = worker_expected
    actual = run('docker', 'image', 'inspect', '--format', '{{.Id}}', image, capture=True)
    if actual != image:
        raise RuntimeError('Local Docker image identity did not match.')
    try:
        worker_image = remote(args.worker, 'docker', 'image', 'inspect', '--format', '{{.Id}}', worker_expected, capture=True)
    except subprocess.CalledProcessError:
        worker_image = ''
    if worker_image != worker_expected:
        if args.command == 'check':
            raise RuntimeError('Worker image is missing; check never transfers or builds images.')
        if args.worker_image:
            raise RuntimeError('The separately qualified worker image is missing; no replacement was made.')
        # Stream the built image; never pull an unrelated mutable registry tag.
        save = subprocess.Popen(['docker', 'save', image], stdout=subprocess.PIPE)
        try:
            load = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.worker, 'docker load'], stdin=save.stdout)
            save.stdout.close()
            if save.wait() or load.returncode:
                raise RuntimeError('Image transfer failed.')
        finally:
            if save.poll() is None:
                save.terminate()
        if remote(args.worker, 'docker', 'image', 'inspect', '--format', '{{.Id}}', image, capture=True) != image:
            raise RuntimeError('Worker image identity did not match after transfer.')
    recipes = cache / 'recipes'
    recipes.mkdir(exist_ok=True)
    path = recipes / (recipe_id + '.json')
    path.write_text(json.dumps(recipe, indent=2) + '\n')
    remote(args.worker, 'mkdir', '-p', str(recipes), str(model_root))
    run('scp', str(path), args.worker + ':' + str(path))
    if args.command == 'check':
        run('lloom', 'setup', '--recipe', recipe_id, '--recipes-root', str(recipes),
            '--model-root', str(model_root), '--additive', '--no-auto-host', '--json')
        return
    # LLooM's pinned downloader reuses its existing local model directories.
    run('lloom', 'install', recipe_id, '--recipes-root', str(recipes), '--model-root', str(model_root), '--apply', '--yes')
    for step in recipe['setup']['steps']:
        if step.get('action') == 'download-model':
            directory = model_root / step['model'].replace('/', '--')
            remote(args.worker, 'mkdir', '-p', str(directory))
            run('rsync', '-a', '--partial', str(directory) + '/', args.worker + ':' + str(directory) + '/')
    # Stop the old distributed runtime before replacing its managed definition.
    installed = json.loads(run('lloom', 'runtime-status', '--json', capture=True))
    if runtime_id in installed.get('runtimes', {}):
        run('lloom', 'runtime-stop', runtime_id, '--json')
    # No direct docker-run, restart policy, aliases, or keep-warm changes.
    run('lloom', 'setup', '--recipe', recipe_id, '--recipes-root', str(recipes),
        '--model-root', str(model_root), '--additive', '--no-auto-host', '--apply', '--yes', '--json')
    run('lloom', 'runtime-start', runtime_id, '--json')
    run('lloom', 'runtime-status', runtime_id, '--json')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
