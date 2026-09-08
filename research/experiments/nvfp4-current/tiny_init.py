# SPDX-License-Identifier: Apache-2.0
"""Deterministic valid FP4 bytes/scales, only for the guarded dummy fixture."""
import hashlib
import torch
from vllm.model_executor.model_loader import weight_utils

_original=weight_utils.initialize_dummy_weights

def initialize(model,model_config,*args,**kwargs):
    if getattr(model_config.hf_config,'_sparkglm_fixture',None)!='tinyglm-nvfp4-v1':
        raise RuntimeError('NVFP4 dummy initialization requires the synthetic fixture')
    _original(model,model_config,*args,**kwargs)
    # vLLM deliberately leaves integer dummy tensors uninitialized. Packed FP4
    # payloads must be initialized to make graph and repeated-output tests valid.
    with torch.no_grad():
        for name,param in model.named_parameters():
            if param.dtype==torch.uint8:
                seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
                g=torch.Generator(device=param.device);g.manual_seed(seed)
                param.copy_(torch.randint(0,256,param.shape,device=param.device,dtype=torch.uint8,generator=g))
            elif param.ndim==1 and param.numel()==4096 and name.endswith('weight'):
                # Keep normalized activations above FP4 scale underflow in the
                # fixture, so routed experts do meaningful work.
                param.fill_(1.0)
            elif 'scale' in name and torch.is_floating_point(param):
                param.fill_(0.015625 if 'weight_scale' in name and not name.endswith('_2') else 1.0)

weight_utils.initialize_dummy_weights=initialize
# The existing image may already have imported this alias via another site hook.
from vllm.model_executor.model_loader import dummy_loader
dummy_loader.initialize_dummy_weights=initialize
