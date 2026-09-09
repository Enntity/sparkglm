#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the default recipe against measured evidence and source availability."""
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('appliance', ROOT / 'scripts/appliance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
image = 'sha256:' + 'a' * 64
for skip in (False, True):
    recipe = module.materialize(image, skip)
    settings = recipe['models'][0]['settings']
    assert settings['contextWindow'] == 524288
    assert settings['maxActiveRequests'] == 4
    assert not settings['keepWarm']
    for member in settings['placement']['members']:
        boot = member['runtimeSettings']['bootstrap']
        assert member['runtimeSettings']['management'] == 'managed'
        assert boot['image'] == image and not boot['pull']
        env = dict(x.split('=', 1) for x in boot['createArgs'] if '=' in x and not x.startswith('type='))
        for key, value in {'MAX_MODEL_LEN':'524288', 'KV_CACHE_MEMORY_BYTES':'9663676416',
                           'MAX_NUM_BATCHED_TOKENS':'2048', 'DFLASH_DRAFT_TP':'2',
                           'DFLASH_TOKENS':'7', 'MOE_BACKEND':'flashinfer_cutlass',
                           'GLM53_MIXED_PREFILL_CHUNK':'skip' if skip else '0'}.items():
            assert env[key] == value, (key, env.get(key))
        assert 'SPARKGLM_EXL3_E3' not in env
try:
    module.materialize('mutable-tag')
    raise AssertionError('mutable image accepted')
except ValueError:
    pass
manifest = json.loads((ROOT / 'profiles/build.json').read_text())
for layer in manifest['layers']:
    subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', layer['revision'] + ':' + layer['dockerfile']], check=True)
plan = subprocess.check_output([str(ROOT / 'start.sh'), 'plan'], text=True)
assert json.loads(plan)['scheduling'] == 'mixed'
print('NVFP4 default: immutable layers, both-rank measured recipe and read-only plan PASS')

exl3 = module.materialize(image, profile='exl3')
assert exl3['models'][0]['settings']['contextWindow'] == 1000000
for member in exl3['models'][0]['settings']['placement']['members']:
    args = member['runtimeSettings']['bootstrap']['createArgs']
    for expected in ('SPARKGLM_EXL3_E3=1', 'SPARKGLM_EXL3_E3_POLICY=concurrent', 'EXL3_TEMP_ROWS_FUSED=32', 'MAX_NUM_BATCHED_TOKENS=7168'):
        assert expected in args, expected

# ModelOpt must remain isolated from the compressed-tensors default.
nvidia = module.materialize(image, profile='nvfp4-nvidia')
assert nvidia['models'][0]['gatewayModel'] == 'sparkglm-nvfp4-nvidia'
assert nvidia['models'][0]['settings']['port'] == 8892
for member in nvidia['models'][0]['settings']['placement']['members']:
    args = member['runtimeSettings']['bootstrap']['createArgs']
    assert 'QUANTIZATION=modelopt_fp4' in args
    assert 'MODEL_DIR=/models/nvidia--GLM-5.3-Flash-NVFP4' in args
    assert 'QUANTIZATION=compressed-tensors' not in args
assert module.materialize(image)['models'][0]['model'].startswith('RedHatAI/')
