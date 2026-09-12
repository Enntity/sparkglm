# SPDX-License-Identifier: Apache-2.0
"""Adapt SparkGLM's deterministic packed-byte initializer to ModelOpt fixtures."""
import hashlib
import torch
from vllm.model_executor.model_loader import weight_utils, dummy_loader

original = weight_utils.initialize_dummy_weights


def initialize(model, model_config, *args, **kwargs):
    if getattr(model_config.hf_config, '_sparkglm_fixture', None) != 'tinyglm-modelopt-v1':
        raise RuntimeError('ModelOpt synthetic initialization requires its fixture marker')
    original(model, model_config, *args, **kwargs)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype == torch.uint8:
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
                generator = torch.Generator(device=param.device).manual_seed(seed)
                param.copy_(torch.randint(0, 256, param.shape, device=param.device,
                                          dtype=torch.uint8, generator=generator))
            elif param.ndim == 1 and param.numel() == 4096 and name.endswith('weight'):
                param.fill_(1.0)
            elif 'scale' in name and torch.is_floating_point(param):
                param.fill_(0.015625 if 'weight_scale' in name and not name.endswith('_2') else 1.0)


weight_utils.initialize_dummy_weights = initialize
dummy_loader.initialize_dummy_weights = initialize
