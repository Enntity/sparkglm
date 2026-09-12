# SPDX-License-Identifier: AGPL-3.0-only
"""Preparation-inclusive KDA screen. Mutable inputs reset outside timed regions."""
import ctypes as C
import json
import math
import statistics
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
P, I = C.c_void_p, C.c_int
lib = C.CDLL(str(ROOT / 'libprobe.so'))
flash = C.CDLL(str(ROOT / 'libatlas_glm53_flash_kda.so'))
base = lib.atlas_probe_baseline
base.argtypes = [P]*11 + [I, I, P]
prep = lib.atlas_probe_prepare
prep.argtypes = [P]*6 + [I, I, P]
trans = lib.atlas_probe_transpose
trans.argtypes = [P, P, I, P]
run = flash.atlas_flash_kda_prefill_fp32_state
run.argtypes = [P]*12 + [I]*4 + [C.c_float]*2 + [P]
size = flash.atlas_flash_kda_workspace_size
size.argtypes, size.restype = [I]*3, C.c_longlong
stream = torch.cuda.current_stream().cuda_stream
report = {'device': torch.cuda.get_device_name(), 'cases': [], 'timing': []}

def check(code):
    if code: raise RuntimeError(f'CUDA launch error {code}')

def ptrs(*xs): return [x.data_ptr() for x in xs]

def rand(shape, dtype=torch.bfloat16, scale=1.):
    return (torch.randn(shape, device='cuda', dtype=torch.float32)*scale).to(dtype)

class Case:
    def __init__(self, t, h, seed=103, degenerate=False):
        torch.manual_seed(seed)
        self.t, self.h = t, h
        self.qkv = rand((t,3,h,128))
        self.gate = rand((t,h,128), scale=2.)
        self.rawbeta = rand((t,h), scale=2.)
        self.a = rand((h,), torch.float32, .2)
        self.bias = rand((h,128), torch.float32, .5)
        self.initial = rand((h,128,128), torch.float32, .1)
        if degenerate:
            self.qkv[:2,:2] = 0
            self.qkv[2:4,:2] *= 1.e-5
            self.gate[::3] = -40
            self.gate[1::3] = 40
            self.rawbeta[::2] = -40
            self.rawbeta[1::2] = 40
        self.s0, self.s1 = self.initial.clone(), self.initial.clone()
        self.st = torch.empty_like(self.s1)
        self.qn, self.kn, self.decay = [torch.empty_like(self.gate, dtype=torch.float32) for _ in range(3)]
        self.beta = torch.empty_like(self.rawbeta, dtype=torch.float32)
        self.q, self.k, self.v = [torch.empty_like(self.gate) for _ in range(3)]
        self.bht = torch.empty((h,t), device='cuda', dtype=torch.bfloat16)
        self.o0, self.o1 = [torch.empty_like(self.gate) for _ in range(2)]
        self.workspace = torch.empty(size(t,h,1), device='cuda', dtype=torch.uint8)
        self.cu = torch.tensor([0,t], device='cuda', dtype=torch.int64)
        self.slots = torch.zeros(1, device='cuda', dtype=torch.int32)
    def baseline(self):
        check(base(*ptrs(self.qkv,self.gate,self.rawbeta,self.a,self.bias,self.qn,self.kn,self.decay,self.beta,self.s0,self.o0),self.t,self.h,stream))
    def candidate(self):
        check(prep(*ptrs(self.qkv,self.rawbeta,self.q,self.k,self.v,self.bht),self.t,self.h,stream))
        check(trans(*ptrs(self.s1,self.st),self.h,stream))
        check(run(*ptrs(self.q,self.k,self.v,self.gate,self.bht,self.st,self.o1,self.workspace,self.a,self.bias,self.cu,self.slots),self.t,self.h,1,1,1/math.sqrt(128),-5.,stream))
        check(trans(*ptrs(self.st,self.s1),self.h,stream))
    def reset(self):
        self.s0.copy_(self.initial)
        self.s1.copy_(self.initial)

def metrics(x,y):
    a,b=x.double(),y.double()
    d=(a-b).abs()
    return {'max_abs': d.max().item(), 'rrmse': (d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)).item(),
            'finite': bool(torch.isfinite(a).all()), 'allclose_1pct': bool(torch.allclose(a,b,atol=.01,rtol=.01))}

def oracle(c):
    q,k,v = c.qkv.double().unbind(1)
    q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)/math.sqrt(128)
    k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    decay=torch.exp(-5*torch.sigmoid(torch.exp(c.a.double())[None,:,None]*(c.gate.double()+c.bias.double())))
    beta=torch.sigmoid(c.rawbeta.double())
    s=c.initial.double().clone()
    out=[]
    for i in range(c.t):
        s=s*decay[i,:,:,None]
        residual=(v[i]-torch.einsum('hkv,hk->hv',s,k[i]))*beta[i,:,None]
        s=s+k[i,:,:,None]*residual[:,None,:]
        out.append(torch.einsum('hkv,hk->hv',s,q[i]))
    return torch.stack(out),s

def assess(t,h,degenerate=False):
    c=Case(t,h,degenerate=degenerate)
    c.baseline();c.candidate();torch.cuda.synchronize()
    item={'tokens':t,'heads':h,'degenerate':degenerate,
          'output':metrics(c.o1,c.o0),'state':metrics(c.s1,c.s0)}
    # Exact transpose involution and unchanged raw operands.
    check(trans(*ptrs(c.initial,c.st),h,stream))
    check(trans(*ptrs(c.st,c.s1),h,stream))
    assert torch.equal(c.initial,c.s1), 'transpose roundtrip'
    assert torch.equal(c.v,c.qkv[:,2]), 'V packing'
    assert torch.equal(c.q,c.qkv[:,0]) and torch.equal(c.k,c.qkv[:,1]), 'raw QK packing'
    assert torch.equal(c.bht,c.rawbeta.T), 'raw beta transpose'
    if t<=33:
        o,s=oracle(c)
        item['baseline_oracle_output']=metrics(c.o0,o)
        item['baseline_oracle_state']=metrics(c.s0,s)
        item['flash_oracle_output']=metrics(c.o1,o)
        c.reset();c.candidate();torch.cuda.synchronize()
        item['flash_oracle_state']=metrics(c.s1,s)
        assert all(item[k]['allclose_1pct'] for k in ['baseline_oracle_output','baseline_oracle_state','flash_oracle_output','flash_oracle_state']), 'independent oracle gate'
    report['cases'].append(item)
    print(json.dumps(item),flush=True)
    assert item['output']['allclose_1pct'] and item['state']['allclose_1pct'], 'numerical gate'
    return c

def timed(c):
    samples={'baseline':[],'flash_inclusive':[]}
    for i in range(18):
        for name in (['baseline','flash_inclusive'] if i%2==0 else ['flash_inclusive','baseline']):
            c.reset()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            (c.baseline if name=='baseline' else c.candidate)()
            end.record();end.synchronize()
            if i>=3: samples[name].append(start.elapsed_time(end))
    item={'tokens':c.t,'heads':c.h,'samples_ms':samples,
          'baseline_ms':statistics.median(samples['baseline']),
          'flash_inclusive_ms':statistics.median(samples['flash_inclusive']),
          'workspace_bytes':c.workspace.numel()}
    item['speedup']=item['baseline_ms']/item['flash_inclusive_ms']
    report['timing'].append(item);print(json.dumps(item),flush=True)

def split_resume(t=4100,cut=4096):
    c=Case(t,32)
    c.baseline();c.candidate()
    oneout,onestate=c.o1.clone(),c.s1.clone()
    state=c.initial.clone()
    parts=[]
    for lo,hi in [(0,cut),(cut,t)]:
        x=Case(hi-lo,32)
        x.qkv=c.qkv[lo:hi].contiguous();x.gate=c.gate[lo:hi].contiguous();x.rawbeta=c.rawbeta[lo:hi].contiguous()
        x.a,x.bias=c.a,c.bias;x.s1.copy_(state)
        x.candidate();state.copy_(x.s1);parts.append(x.o1.clone())
    out=torch.cat(parts)
    # Resume both final states through three real current recurrence steps.
    z=Case(3,32,seed=907)
    z.s0.copy_(c.s0);z.baseline();refout=z.o0.clone();refstate=z.s0.clone()
    z.s0.copy_(state);z.baseline();torch.cuda.synchronize()
    item={'tokens':t,'split_at':cut,'output':metrics(out,c.o0),'state':metrics(state,c.s0),
          'flash_split_output':metrics(out,oneout),'flash_split_state':metrics(state,onestate),
          'resume_output':metrics(z.o0,refout),'resume_state':metrics(z.s0,refstate)}
    report.setdefault('split_resume',[]).append(item);print(json.dumps(item),flush=True)
    assert all(item[k]['allclose_1pct'] for k in ['output','state','flash_split_output','flash_split_state','resume_output','resume_state'])

if __name__ == '__main__':
 try:
    for t in [1,17,33]: assess(t,2,True)
    for t in [2048,4096,4100]:
        c=assess(t,32);timed(c)
    split_resume(cut=4096)
    split_resume(t=2051,cut=2035)
    report['qualified_for_model_screen']=all(x['speedup']>=1.5 for x in report['timing'])
 except Exception as e:
    report['error']=repr(e)
    raise
 finally:
    (ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
