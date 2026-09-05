# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['installed', 'loaded', 'jit-source'], default='installed')
p.add_argument('--paths')
a = p.parse_args()
site = Path('/usr/local/lib/python3.12/dist-packages')
report = {'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions(path=[str(site)])}}
report['environment'] = {k: v for k, v in os.environ.items() if k in ['LD_LIBRARY_PATH', 'PYTHONPATH', 'CUDA_VERSION', 'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'FLASHINFER_CUDA_ARCH_LIST']}
paths = set()
if a.mode == 'loaded':
    report['processes'] = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if not ('VLLM::Worker' in cmd or 'VLLM::Engine' in cmd):
                continue
            report['processes'].append({'pid': proc.name, 'role': cmd.split()[0]})
            for line in (proc / 'maps').read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and parts[5].startswith('/') and '.so' in parts[5] and not parts[5].endswith(' (deleted)'):
                    paths.add(Path(parts[5]))
        except (FileNotFoundError, PermissionError):
            pass
elif a.mode == 'jit-source':
    for root in [site / 'vllm/third_party/deep_gemm', site / 'vllm/include', site / 'triton']:
        if root.exists():
            paths.update(x for x in root.rglob('*') if x.is_file() and x.suffix in ['.h', '.hpp', '.cuh', '.cu', '.cpp', '.cc', '.json', '.py'])
elif a.paths:
    paths = {Path(x) for x in json.loads(Path(a.paths).read_text())}
else:
    for root in [site, Path('/usr/local/cuda/targets/sbsa-linux/lib'), Path('/opt/glm53'), Path('/opt/sparkglm')]:
        if root.exists():
            paths.update(x for x in root.rglob('*') if x.is_file() and ('.so' in x.name or x.suffix in ['.cubin', '.ptx'] or x.name == 'sitecustomize.py'))
memo = {}
report['files'] = {}
for path in sorted(paths):
    if not path.is_file():
        report['files'][str(path)] = {'missing': True}
        continue
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino)
    if key not in memo:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(block)
        memo[key] = h.hexdigest()
    report['files'][str(path)] = {'sha256': memo[key], 'bytes': stat.st_size, 'resolved': str(path.resolve())}
print(json.dumps(report, sort_keys=True))
