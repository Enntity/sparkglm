#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Protect standalone ownership and measured serving arguments without a gateway."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import standalone
from appliance import materialize

image='sha256:'+'a'*64
config=dict(worker='worker',head_address='192.0.2.1',worker_address='192.0.2.2',
            model_root='/models',entrypoint='/runtime/entrypoint.sh',interfaces=['eth0','eth1'],
            hcas=['rdma0','rdma1'],gids=[3,4],images=[image,image],names=['test-head','test-worker'])
for profile in ('nvfp4','exl3'):
    recipe=materialize(image,profile=profile)
    for rank in (0,1):
        args=standalone.docker_args(recipe,config,rank)
        assert not any('lloom' in x or '${' in x for x in args), args
        assert args[-2:]==[image,'/opt/sparkglm/entrypoint.sh']
        assert standalone.OWNER in args
        assert 'NODE_RANK='+str(rank) in args
        assert 'NCCL_IB_GID_INDEX='+str(config['gids'][rank]) in args
        assert 'VLLM_HOST='+config['head_address' if rank==0 else 'worker_address'] in args
        for member in recipe['models'][0]['settings']['placement']['members']:
            for value in member['runtimeSettings']['bootstrap']['createArgs']:
                if value.startswith(('MAX_MODEL_LEN=','KV_CACHE_MEMORY_BYTES=','MAX_NUM_BATCHED_TOKENS=',
                                     'DFLASH_','MOE_BACKEND=','GLM53_MIXED_PREFILL_CHUNK=')):
                    assert value in args,value
calls=[]
def foreign(config,rank,*args,capture=False):
    calls.append(args)
    return '' if 'label='+standalone.OWNER in args else 'foreign-container'
with patch.object(standalone,'command',foreign):
    try:
        standalone.containers(config,'remove')
        raise AssertionError('foreign container accepted')
    except RuntimeError:
        pass
assert all('rm' not in x and 'stop' not in x for x in calls)
plan=json.loads(subprocess.check_output([str(ROOT/'start.sh'),'plan'],text=True))
assert plan['manager']=='standalone'
assert plan['profile']['models'][0]['gatewayModel']=='sparkglm-nvfp4'
optional=json.loads(subprocess.check_output([str(ROOT/'start.sh'),'--lloom','plan'],text=True))
assert optional['profile']['models'][0]['gatewayModel']=='sparkglm-nvfp4'
assert (ROOT/'runtime/entrypoint.sh').read_bytes().startswith(b'#!/usr/bin/env bash')
print('Standalone command parity, ownership isolation and optional LLooM dispatch PASS')
