#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Independent synthetic byte oracle. No timing; parent authorizes GPU run."""
import ctypes as C
import hashlib
import json
from pathlib import Path

def main():
    import torch
    def need(ok,msg):
        if not ok: raise AssertionError(msg)
    def report(**kw): print(json.dumps(kw,allow_nan=False),flush=True)
    path=Path('/eval/libatlas_native_prep.so')
    report(stage='library',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    limit=1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.,limit/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(); torch.set_grad_enabled(False); torch.manual_seed(20260911)
    dev='cuda:0'; stream=torch.cuda.current_stream().cuda_stream
    lib=C.CDLL(str(path)); lib.atlas_native_prep_version.restype=C.c_int
    lib.atlas_native_prep_version.argtypes=[]
    need(lib.atlas_native_prep_version()==1,'ABI mismatch')
    qop=lib.atlas_native_prep_q; qop.restype=C.c_int
    qop.argtypes=[C.c_void_p,C.c_void_p,C.c_uint32,C.c_uint64,C.c_uint64,C.c_void_p]
    kop=lib.atlas_native_prep_kv; kop.restype=C.c_int
    kop.argtypes=[C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint64,C.c_uint32,C.c_void_p]
    def equal(a,b,label):
        x,y=a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)
        same=x==y
        if not bool(same.all()):
            idx=torch.nonzero(~same,as_tuple=False)[0].tolist()
            raise AssertionError(f'{label}: first unequal byte {idx}, actual={int(x[tuple(idx)])}, expected={int(y[tuple(idx)])}')
    def guard(n):
        raw=torch.full((n+512,),0xa5,dtype=torch.uint8,device=dev)
        return raw,raw[256:-256]
    def intact(raw):
        need(bool((raw[:256]==0xa5).all()) and bool((raw[-256:]==0xa5).all()),'redzone modified')
    for rows in [1,3,65,128]:
        q=torch.randn((rows,32,512),device=dev).bfloat16()
        bits=q.view(torch.int16).flatten()
        bits[:6]=torch.tensor([0,-32768,32705,-63,1,-32767],dtype=torch.int16,device=dev)
        saved=q.clone(); raw,body=guard(rows*32*576*2)
        out=body.view(torch.bfloat16).reshape(rows,32,576)
        args=[q.data_ptr(),out.data_ptr(),rows,q.numel()*2,body.numel(),stream]
        if rows==1:
            for i,v,code in [(0,None,-1),(2,0,-2),(3,q.numel()*2-1,-3),(4,body.numel()-1,-3),(1,q.data_ptr(),-4)]:
                bad=args.copy();bad[i]=v;need(qop(*bad)==code,'Q rejection failed')
            need(bool((raw==0xa5).all()),'Q rejection wrote output')
        need(qop(*args)==0,'Q launch failed');torch.cuda.synchronize()
        expected=torch.zeros_like(out);expected[...,:512]=saved
        equal(out,expected,'Q copy/zero');equal(q,saved,'Q input');intact(raw)
        report(stage='q',rows=rows,passed=True)
    for seq in [1,63,64,65,4096,32768]:
      for mode in ['zeros','tiny','random','extreme']:
        nb=(seq+15)//16; pb=nb+3; padded=(seq+63)//64*64
        table=torch.arange(nb+1,1,-1,device=dev,dtype=torch.int32)
        src=torch.randn((pb,16,512),device=dev).bfloat16()
        if mode=='zeros': src.zero_()
        if mode=='tiny': src.mul_(1e-38)
        if mode=='extreme':
            values=torch.tensor([0.,-0.,1e-40,-1e-40,1e-4,-1e-4,torch.finfo(torch.bfloat16).max,-torch.finfo(torch.bfloat16).max],device=dev).bfloat16()
            src.reshape(-1).copy_(values.repeat(src.numel()//8))
        saved=src.clone(); saved_table=table.clone(); raw,body=guard(padded*656)
        args=[src.data_ptr(),pb,table.data_ptr(),nb,body.data_ptr(),body.numel(),seq,stream]
        if seq==1 and mode=='zeros':
            for i,v,code in [(0,None,-1),(1,0,-2),(2,None,-1),(3,0,-3),(5,body.numel()-1,-3),(6,0,-2),(4,src.data_ptr(),-4)]:
                bad=args.copy();bad[i]=v;need(kop(*bad)==code,'KV rejection failed')
            need(bool((raw==0xa5).all()),'KV rejection wrote output')
        need(kop(*args)==0,'KV launch failed');torch.cuda.synchronize()
        logical=saved.index_select(0,table.long()).reshape(-1,512)[:seq].float().reshape(seq,4,128)
        scales=logical.abs().amax(-1).clamp_min(1e-4)/448.0
        quant=(logical/scales[...,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        expected=torch.zeros((padded,656),dtype=torch.uint8,device=dev)
        expected[:seq,:512]=quant.contiguous().view(torch.uint8).reshape(seq,512)
        expected[:seq,512:528]=scales.contiguous().view(torch.uint8).reshape(seq,16)
        expected[seq:,512:528]=torch.ones((padded-seq,4),device=dev).view(torch.uint8)
        equal(body.reshape(padded,656),expected,f'KV seq={seq} mode={mode}')
        equal(src,saved,'KV input');equal(table,saved_table,'block table');intact(raw)
        need(torch.cuda.max_memory_reserved()<=limit,'memory cap exceeded')
        report(stage='kv',seqend=seq,mode=mode,passed=True)
        del src,saved,logical,quant,expected,raw,body,scales
    report(stage='complete',passed=True,q_cases=4,kv_cases=24,peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='preparation byte oracle only; no attention or model correctness claim')

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(json.dumps({'stage':'failure','passed':False,'error':str(e)}),flush=True)
        raise SystemExit(1)
