#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Original clamped W4A4 API probe, not a model benchmark or quality claim."""
import argparse
import json
import torch
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe

torch.manual_seed(20260907)
device='cuda'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rows',type=int,default=64)
p.add_argument('--experts',type=int,default=8)
p.add_argument('--graph',action='store_true')
a=p.parse_args()
if a.rows<1 or not 8<=a.experts<=32: p.error('positive rows and 8..32 experts required')
m,h,i,e,k=a.rows,4096,1024,a.experts,8
x=torch.randn(m,h,device=device,dtype=torch.bfloat16)*0.2
gate=torch.randn(e,i,h,device=device,dtype=torch.bfloat16)*0.1
up=torch.randn_like(gate)*0.1
down=torch.randn(e,h,i,device=device,dtype=torch.bfloat16)*0.05
ones=torch.ones(e,device=device,dtype=torch.float32)
gs=torch.ones(1,device=device,dtype=torch.float32)
# FlashInfer's fused MoE API stores up rows followed by gate rows.
w13=torch.cat((up,gate),1)
q13,s13=fp4_quantize(w13.flatten(0,1),global_scale=gs,sf_vec_size=16,is_sf_swizzled_layout=True)
q2,s2=fp4_quantize(down.flatten(0,1),global_scale=gs,sf_vec_size=16,is_sf_swizzled_layout=True)
qx,sx=fp4_quantize(x,global_scale=gs,sf_vec_size=16,is_sf_swizzled_layout=True)
ids=(torch.arange(k,device=device,dtype=torch.int32)[None,:]+torch.arange(m,device=device,dtype=torch.int32)[:,None])%e
weights=torch.full((m,k),1/k,device=device,dtype=torch.float32)
limit=torch.full((e,),1.0,device=device,dtype=torch.float32)
output=torch.empty_like(x)
def run():
    return cutlass_fused_moe(input=qx,input_sf=sx,
        token_selected_experts=ids,token_final_scales=weights,
        fc1_expert_weights=q13.view(e,2*i,h//2).view(torch.long),
        fc2_expert_weights=q2.view(e,h,i//2).view(torch.long),
        output=output,output_dtype=torch.bfloat16,
        quant_scales=[ones,s13.view(e,2*i,-1).view(torch.int32),ones,
                      ones,s2.view(e,h,-1).view(torch.int32),ones],
        swiglu_limit=limit,tp_size=1,tp_rank=0,ep_size=1,ep_rank=0)
run();torch.cuda.synchronize()
ref=torch.zeros_like(x,dtype=torch.float32)
for n in range(e):
    g=(x.float()@gate[n].float().T).clamp(max=1)
    u=(x.float()@up[n].float().T).clamp(min=-1,max=1)
    selected=(ids==n).any(dim=1).float()[:,None]
    ref.add_(((torch.nn.functional.silu(g)*u)@down[n].float().T)*selected,alpha=1/k)
assert torch.isfinite(output).all()
# A second reference includes the actual W4A4 round trips. Obtain unswizzled
# scales independently for scalar decoding, rather than reverse-engineering the
# kernel's scale-memory layout. Packed E2M1 values use low nibble first.
def roundtrip(t):
    q,s=fp4_quantize(t.contiguous(),global_scale=gs,sf_vec_size=16,is_sf_swizzled_layout=False)
    codes=torch.stack((q&15,q>>4),dim=-1).reshape(t.shape)
    table=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device=device)
    if s.dtype==torch.uint8: s=s.view(torch.float8_e4m3fn)
    scales=s.flatten()[:t.numel()//16].reshape(t.shape[0],t.shape[1]//16).float().repeat_interleave(16,dim=1)
    return table[codes.long()]*scales
xd=roundtrip(x)
exact=torch.zeros_like(ref)
for n in range(e):
    wd=roundtrip(w13[n])
    projections=(xd@wd.T).to(torch.bfloat16).float()
    u,g=projections.chunk(2,dim=-1)
    activated=(torch.nn.functional.silu(g.clamp(max=1))*u.clamp(-1,1)).to(torch.bfloat16)
    ad=roundtrip(activated)
    partial=(ad@roundtrip(down[n]).T).to(torch.bfloat16).float()
    exact.add_(partial*(ids==n).any(dim=1).float()[:,None],alpha=1/k)
exact_relative=float(torch.linalg.vector_norm(output.float()-exact)/torch.linalg.vector_norm(exact))
clamped=output.clone()
limit.fill_(100000)
run();torch.cuda.synchronize()
clamp_effect=float(torch.linalg.vector_norm(output.float()-clamped)/torch.linalg.vector_norm(clamped.float()))
assert clamp_effect>0.1, 'clamp has no material effect'
limit.fill_(1)
run();torch.cuda.synchronize()
relative=float(torch.linalg.vector_norm(output.float()-ref)/torch.linalg.vector_norm(ref))
cosine=float(torch.nn.functional.cosine_similarity(output.float().flatten(),ref.flatten(),dim=0))
graph_pass=None
if a.graph:
    expected=output.clone()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(output,expected,rtol=0.03,atol=0.01)
    packed=qx.clone();qx.zero_()
    graph.replay();torch.cuda.synchronize()
    assert not torch.count_nonzero(output), 'changed-input graph replay retained stale output'
    qx.copy_(packed)
    graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(output,expected,rtol=0.03,atol=0.01)
    graph_pass=True
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
    run();torch.cuda.synchronize()
names=sorted(set(v.name for v in p.events() if v.device_type==torch.autograd.DeviceType.CUDA))
print(json.dumps(dict(gpu=torch.cuda.get_device_name(),rows=m,experts=e,top_k=k,graph_replay=graph_pass,relative_l2=relative,cosine=cosine,quantized_reference_relative_l2=exact_relative,clamp_effect_relative_l2=clamp_effect,cuda_kernels=names),indent=2))
if relative>0.3 or cosine<0.96 or exact_relative>0.03: raise SystemExit('clamped W4A4 numerical screen failed')
