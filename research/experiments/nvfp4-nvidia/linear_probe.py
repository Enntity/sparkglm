#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ModelOpt dense W4A4 at GLM TP2 dimensions against decoded FP4 operands."""
import json
import tempfile
import torch
from flashinfer.fp4_quantization import fp4_quantize
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized

torch.manual_seed(20260909)
torch.set_default_device('cuda')
rendezvous = tempfile.TemporaryDirectory()
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method='file://' + rendezvous.name + '/store')
scale = torch.ones(1, dtype=torch.float32)


def quantize(x):
    packed, scales = fp4_quantize(x.contiguous(), global_scale=scale,
                                  sf_vec_size=16, is_sf_swizzled_layout=False)
    if scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e4m3fn)
    scales = scales.flatten()[:x.numel() // 16].reshape(x.shape[0], x.shape[1] // 16)
    return packed, scales


def decode(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    return table[codes.long()] * scales.float().repeat_interleave(16, dim=1)


records = []
with set_current_vllm_config(VllmConfig()):
    ensure_model_parallel_initialized(1, 1)
    config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
    for n, k in ((12288, 4096), (4096, 6144)):
        method = config.LinearMethodCls(config)
        layer = torch.nn.Module()
        method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.bfloat16) * .03
        packed, scales = quantize(weight)
        reference_weight = decode(packed, scales)
        with torch.no_grad():
            layer.weight.copy_(packed)
            layer.weight_scale.copy_(scales)
            layer.input_scale.fill_(1)
            layer.weight_scale_2.fill_(1)
        method.process_weights_after_loading(layer)
        for rows in (1, 2, 4, 32, 2048):
            x = torch.randn(rows, k, dtype=torch.bfloat16) * .2
            xp, xs = quantize(x)
            expected = decode(xp, xs) @ reference_weight.T
            actual = method.apply(layer, x)
            torch.cuda.synchronize()
            relative = float(torch.linalg.vector_norm(actual.float() - expected) /
                             torch.linalg.vector_norm(expected))
            assert torch.isfinite(actual).all() and relative < .03, (rows, n, k, relative)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = method.apply(layer, x)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, actual, rtol=.01, atol=.01)
            x.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert not torch.count_nonzero(captured), 'graph retained stale inputs'
            records.append(dict(rows=rows, n=n, k=k, relative_l2=relative,
                                backend=type(method.kernel).__name__, graph_replay=True))
print(json.dumps(records, indent=2))
