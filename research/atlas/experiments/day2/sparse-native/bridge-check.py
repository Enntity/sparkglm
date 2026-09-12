#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline C ABI versus retained TVM-FFI byte parity; parent owns GPU isolation."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path


def main():
    import torch
    import tvm_ffi
    def require(ok, why):
        if not ok: raise AssertionError(why)
    def report(**kw): print(json.dumps(kw, allow_nan=False), flush=True)
    paths = [Path('/eval/libatlas_sparse_native.so'), Path('/native/sparse_mla_sm120.so')]
    report(stage='source', sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    require(torch.cuda.get_device_capability() == (12, 1), 'expected SM121')
    limit = 1024**3
    torch.cuda.set_per_process_memory_fraction(min(1., limit/torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats(); torch.set_grad_enabled(False); torch.manual_seed(20260911)
    lib = C.CDLL(str(paths[0])); lib.atlas_sparse_native_version.restype = C.c_int
    lib.atlas_sparse_native_version.argtypes = []
    require(lib.atlas_sparse_native_version() == 1, 'C ABI version mismatch')
    launch = lib.atlas_sparse_native_launch_glm
    launch.argtypes = [C.c_void_p]*5 + [C.c_int32, C.c_int32, C.c_float, C.c_void_p, C.c_void_p]
    launch.restype = C.c_int
    module = tvm_ffi.load_module(str(paths[1])); op = module.sparse_mla_sm120_paged_attention
    T,H,D,K,S = 128,32,512,2048,4096
    dev = 'cuda:0'; scale = C.c_float(1/math.sqrt(D)).value
    for name in ['singleton','zero_query','random']:
        lens = torch.tensor([1 if name=='singleton' else [1,3,63,64,65,127,128,129,K][r%9] for r in range(T)], dtype=torch.int32, device=dev)
        ids = torch.full((T,K), -1, dtype=torch.int32, device=dev)
        for r, length in enumerate(lens.cpu().tolist()):
            ids[r,:length] = ((torch.arange(length, device=dev)*17 + [0,63,64,127,128,S-1][r%6])%S).int()
        q = torch.zeros((T,H,576), dtype=torch.bfloat16, device=dev)
        if name=='random':
            latent = torch.randn((S,D), device=dev).bfloat16()
            q[...,:D] = torch.randn((T,H,D), device=dev).bfloat16()
        elif name=='singleton':
            latent = (((torch.arange(S,device=dev)[:,None]+torch.arange(D,device=dev)[None,:])%17-8)*.25).bfloat16()
        else: latent = torch.full((S,D), .5, dtype=torch.bfloat16, device=dev)
        x = latent.float().reshape(S,4,128)
        scales = x.abs().amax(-1).clamp_min(1e-4)/448 if name=='random' else torch.ones((S,4),device=dev)
        packed = torch.zeros((S,656), dtype=torch.uint8, device=dev)
        packed[:,:D] = (x/scales[...,None]).clamp(-448,448).to(torch.float8_e4m3fn).contiguous().view(torch.uint8).reshape(S,D)
        packed[:,512:528] = scales.contiguous().view(torch.uint8).reshape(S,16)
        kv = packed.reshape(S//64,64,656)
        ref = torch.full((T,H,D), float('nan'), dtype=torch.bfloat16, device=dev)
        out = torch.full_like(ref,float('nan')); rl = torch.full((T,H),float('nan'),device=dev); ol = torch.full_like(rl,float('nan'))
        args = [q.data_ptr(),kv.data_ptr(),ids.data_ptr(),out.data_ptr(),ol.data_ptr(),T,H,scale,lens.data_ptr(),torch.cuda.current_stream().cuda_stream]
        if name=='singleton':
            poison_out, poison_lse = out.view(torch.uint8).clone(), ol.view(torch.uint8).clone()
            rejects = [(i,None,-1) for i in [0,1,2,3,4,8]] + [(5,64,-2),(5,-1,-2),(6,7,-3),(7,0.,-4),(7,-1.,-4),(7,float('nan'),-4),(7,float('inf'),-4)]
            for index,value,code in rejects:
                bad=args.copy(); bad[index]=value
                require(launch(*bad)==code, f'rejection argument {index} expected {code}')
            require(torch.equal(out.view(torch.uint8),poison_out) and torch.equal(ol.view(torch.uint8),poison_lse), 'rejected call wrote outputs')
            report(stage='rejects', passed=True, cases=len(rejects))
        op(q,kv,ids,ref,rl,scale,2,lens,None,None,None,None)
        require(launch(*args)==0, f'{name}: C ABI dispatch failed')
        torch.cuda.synchronize()
        for label,a,b in [('output',out,ref),('lse',ol,rl)]:
            require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), f'{name}/{label}: nonfinite or unwritten')
            same = a.contiguous().view(torch.uint8)==b.contiguous().view(torch.uint8)
            if not bool(same.all()):
                coord = torch.nonzero(~same, as_tuple=False)[0].tolist()
                raise AssertionError(f'{name}/{label}: first unequal byte coordinate {coord}')
        require(torch.cuda.max_memory_reserved()<=limit, 'Torch memory exceeds 1GiB')
        report(stage=name, passed=True, exact_output_bytes=out.numel()*2, exact_lse_bytes=ol.numel()*4)
    report(stage='complete', passed=True, cases=3, peak_reserved_bytes=torch.cuda.max_memory_reserved(), scope='C ABI versus TVM FFI byte parity only; no timing or independent mathematical qualification')


if __name__=='__main__':
    try: main()
    except Exception as e:
        print(json.dumps({'stage':'failure','passed':False,'error':str(e)}),flush=True)
        raise SystemExit(1)
