#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Protect solo dispatch and avoid device synchronization in E3 selection."""
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch
import os
import sys

policy = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    'research/experiments/exl3-e3/policy.py'))['concurrent_prefill']
meta = lambda d, p: {'attention': SimpleNamespace(num_decodes=d, num_prefills=p)}
assert not policy(None)  # adapter reserves E3 scratch before reference profiling
assert not policy(meta(0, 1))
assert policy(meta(1, 1))
assert policy(meta(0, 2))
assert not policy(meta(4, 0))
assert not policy({})
assert not policy({'attention': object()})
assert not policy([meta(1, 1)])  # unsupported DBO microbatch selection

class DeviceScalar:
    def __int__(self):
        raise AssertionError('dispatch attempted a device synchronization')

assert not policy(meta(DeviceScalar(), 1))

# Exercise the actual adapter without a CUDA dependency. Profiling must reserve
# E3 buffers while returning False so the caller also profiles reference scratch.
state = {'metadata': None, 'capturing': False}
allocations = []
def empty(shape, **kwargs):
    allocations.append(shape)
    return SimpleNamespace(shape=shape)

fake_torch = SimpleNamespace(
    empty=empty, float16='float16',
    cuda=SimpleNamespace(is_current_stream_capturing=lambda: state['capturing']))
modules = {
    'torch': fake_torch,
    'exl3_fat_moe_ext': SimpleNamespace(),
    'sparkglm_e3_tables': SimpleNamespace(build_grouped_fat_tables=None),
    'sparkglm_e3_policy': SimpleNamespace(concurrent_prefill=policy),
    'vllm.forward_context': SimpleNamespace(
        is_forward_context_available=lambda: True,
        get_forward_context=lambda: SimpleNamespace(attn_metadata=state['metadata'])),
}
with patch.dict(sys.modules, modules), patch.dict(os.environ, {'SPARKGLM_EXL3_E3_POLICY': 'concurrent'}):
    runtime = runpy.run_path(str(Path(__file__).resolve().parents[1] /
        'research/experiments/exl3-e3/runtime.py'))
    eligible = runtime['eligible']
    x = SimpleNamespace(device='cuda:0', shape=(8, 4096))
    assert not eligible(x, 64, 1024)
    assert allocations == [(64, 4096), (64, 1024)]
    assert not eligible(x, 64, 1024)
    assert len(allocations) == 2
    state['metadata'] = meta(0, 1)
    assert not eligible(x, 64, 1024)
    state['metadata'] = meta(1, 1)
    assert eligible(x, 64, 1024)
    state['metadata'] = None
    state['capturing'] = True
    try:
        eligible(x, 128, 1024)
    except RuntimeError as exc:
        assert 'before CUDA graph capture' in str(exc)
    else:
        raise AssertionError('E3 scratch grew during graph capture')
    state['capturing'] = False
    os.environ['SPARKGLM_EXL3_E3_POLICY'] = 'large'
    assert not eligible(x, 64, 1024)
    state['metadata'] = meta(0, 1)
    assert eligible(x, 64, 1024)
print('E3 concurrent metadata policy: PASS')
