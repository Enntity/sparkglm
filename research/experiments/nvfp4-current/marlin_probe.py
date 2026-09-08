#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Original NVFP4/Marlin contract probe using the installed vLLM conversion API."""
import argparse
import json
from types import SimpleNamespace
import torch
from flashinfer.fp4_quantization import fp4_quantize
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.scalar_type import scalar_types

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rows',type=int,default=65)
p.add_argument('--experts',type=int,default=16)
p.add_argument('--graph',action='store_true')
a=p.parse_args()
if a.rows<1 or a.experts not in (8,16,288):p.error('positive rows; experts 8, 16, or 288')
torch.manual_seed(20260907)
m,h,i,e,k=a.rows,4096,1024,a.experts,8
device='cuda';dtype=torch.bfloat16
x=torch.randn(m,h,device=device,dtype=dtype)*0.2
w13=torch.randn(e,2*i,h,device=device,dtype=dtype)*0.7
w2=torch.randn(e,h,i,device=device,dtype=dtype)*0.05
gs=torch.ones(1,device=device,dtype=torch.float32)
def quant(t):
 q,s=fp4_quantize(t.flatten(0,1).contiguous(),global_scale=gs,sf_vec_size=16,is_sf_swizzled_layout=False)
 if s.dtype==torch.uint8:s=s.view(torch.float8_e4m3fn)
 return q.reshape(*t.shape[:-1],t.shape[-1]//2),s.flatten()[:t.numel()//16].reshape(*t.shape[:-1],t.shape[-1]//16)
q13,s13=quant(w13);q2,s2=quant(w2)
# Independent scalar decoding of unpermuted E2M1 packed weights and group scales.
def decode(q,s):
 codes=torch.stack((q&15,q>>4),dim=-1).flatten(-2)
 table=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device=device)
 return table[codes.long()]*s.float().repeat_interleave(16,dim=-1)
ids=(torch.arange(k,device=device,dtype=torch.int32)[None,:]+torch.arange(m,device=device,dtype=torch.int32)[:,None])%e
weights=torch.full((m,k),1/k,device=device,dtype=torch.float32)
layer=SimpleNamespace(num_experts=e,hidden_size=h,intermediate_size_per_partition=i,params_dtype=dtype)
b13,bs13,bg13,b2,bs2,bg2=prepare_nvfp4_moe_layer_for_marlin(layer,q13,s13,torch.ones(e,device=device),q2,s2,torch.ones(e,device=device),True)
guarded=torch.full((m*h+256,),float('nan'),device=device,dtype=dtype)
output=guarded[128:-128].view(m,h)
activation=ApplyMoEActivationConfig(clamp_limit=10.)
def run():
 return fused_marlin_moe(hidden_states=x,w1=b13,w2=b2,bias1=None,bias2=None,w1_scale=bs13,w2_scale=bs2,
  topk_weights=weights,topk_ids=ids,quant_type_id=scalar_types.float4_e2m1f.id,
  global_scale1=bg13,global_scale2=bg2,workspace=layer.workspace,output=output,activation_config=activation)
run();torch.cuda.synchronize()
reference=torch.zeros((m,h),device=device,dtype=torch.float32)
for n in range(e):
 selected=(ids==n).any(dim=1)
 # The golden only computes rows routed to this expert; it is not a timed path.
 positions=selected.nonzero().flatten()
 if not positions.numel():continue
 proj=(x[positions].float()@decode(q13[n],s13[n]).T).to(dtype).float()
 gate,up=proj.chunk(2,dim=-1)
 act=(torch.nn.functional.silu(gate.clamp(max=10))*up.clamp(-10,10)).to(dtype)
 partial=(act.float()@decode(q2[n],s2[n]).T).to(dtype).float()/k
 reference.index_add_(0,positions,partial)
relative=float(torch.linalg.vector_norm(output.float()-reference)/torch.linalg.vector_norm(reference))
assert torch.isfinite(output).all() and relative<=0.03, relative
assert torch.isnan(guarded[:128]).all() and torch.isnan(guarded[-128:]).all()
expected=output.clone();activation=ApplyMoEActivationConfig()
run();torch.cuda.synchronize()
clamp_effect=float(torch.linalg.vector_norm(output.float()-expected.float())/torch.linalg.vector_norm(expected.float()))
assert clamp_effect>0.1, 'production clamp has no material effect'
activation=ApplyMoEActivationConfig(clamp_limit=10.)
run();torch.cuda.synchronize()
if a.graph:
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):run()
 graph.replay();torch.cuda.synchronize();torch.testing.assert_close(output,expected,rtol=0.03,atol=0.01)
 saved=x.clone();x.zero_();graph.replay();torch.cuda.synchronize();assert not torch.count_nonzero(output)
 x.copy_(saved);graph.replay();torch.cuda.synchronize();torch.testing.assert_close(output,expected,rtol=0.03,atol=0.01)
assert torch.isnan(guarded[:128]).all() and torch.isnan(guarded[-128:]).all()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
 run();torch.cuda.synchronize()
names=sorted({v.name for v in profiler.events() if v.device_type==torch.autograd.DeviceType.CUDA})
print(json.dumps({'rows':m,'experts':e,'top_k':k,'hidden':h,'intermediate':i,'clamp':10,'quantized_reference_relative_l2':relative,'clamp_effect_relative_l2':clamp_effect,'graph_replay':a.graph,'redzones':True,'cuda_kernels':names},indent=2))
