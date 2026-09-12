#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded unified ABI byte equivalence and inclusive timing; parent runs GPU."""
import argparse
import ctypes as C
import gc
import hashlib
import json
from pathlib import Path
import statistics

class Args(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ['q','kv','selected','table','qpad','packed_kv','metadata','main_out','tail_out','out','metadata_bytes','stream']]+[(n,C.c_uint32) for n in ['rows','seq_start','physical_blocks','block_table_count']]

def require(ok,why):
    if not ok: raise AssertionError(why)
def report(**kw): print(json.dumps(kw,allow_nan=False),flush=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def layout(rows):
    regions={}; offset=0
    for name,shape in [('main_ids',(rows,2048)),('tail_ids',(rows,2048)),('main_lengths',(rows,)),('tail_lengths',(rows,)),('tail_counts',(rows,)),('main_lse',(rows,32)),('tail_lse',(rows,32)),('final_lse',(rows,32))]:
        offset=(offset+255)//256*256
        count=1
        for n in shape: count*=n
        regions[name]=(offset,count*4,shape); offset+=count*4
    return regions,offset

def main(opt):
    require(C.sizeof(Args)==112 and C.alignment(Args)==8,'ABI size/alignment')
    for i,(name,_) in enumerate(Args._fields_):
        require(getattr(Args,name).offset==(i*8 if i<12 else 96+(i-12)*4),'ABI offset '+name)
    import torch
    import tvm_ffi
    cap=2*1024**3; minimum=4*1024**3; dev='cuda:0'
    require(torch.cuda.get_device_capability()==(12,1),'expected SM121')
    free,total=torch.cuda.mem_get_info(); require(free>=minimum,'requires 4GiB free')
    torch.cuda.set_per_process_memory_fraction(min(1.,cap/total),0)
    torch.set_grad_enabled(False); torch.manual_seed(20260911)
    uni=C.CDLL(opt.unified); atlas=C.CDLL(opt.atlas); merge=C.CDLL(opt.merge)
    native=tvm_ffi.load_module(opt.native); op=native.sparse_mla_sm120_paged_attention
    uni.atlas_glm_sparse_native_version.argtypes=[]; uni.atlas_glm_sparse_native_version.restype=C.c_int
    uni.atlas_glm_sparse_native_init.argtypes=[]; uni.atlas_glm_sparse_native_init.restype=C.c_int
    uni.atlas_glm_sparse_native_run.argtypes=[C.POINTER(Args)]; uni.atlas_glm_sparse_native_run.restype=C.c_int
    atlas.atlas_sparse_init.argtypes=[]; atlas.atlas_sparse_init.restype=C.c_int
    atlas.atlas_sparse_run.argtypes=[C.c_void_p]*5+[C.c_uint]*4+[C.c_float,C.c_void_p]; atlas.atlas_sparse_run.restype=C.c_int
    merge.atlas_sparse_merge.argtypes=[C.c_void_p]*7+[C.c_uint]*2+[C.c_void_p]; merge.atlas_sparse_merge.restype=C.c_int
    require(uni.atlas_glm_sparse_native_version()==1,'version mismatch')
    require(uni.atlas_glm_sparse_native_init()==0,'unified init')
    require(atlas.atlas_sparse_init()==0,'Atlas init')
    torch.cuda.synchronize()
    report(stage='init',passed=True,abi_size=C.sizeof(Args),library_sha256={k:sha(v) for k,v in vars(opt).items()},torch=torch.__version__,cuda=torch.version.cuda,cap_bytes=cap,scope='Synthetic integration byte equivalence; no model qualification')
    def mem():
        n=torch.cuda.max_memory_reserved(); require(n<=cap,'2GiB Torch cap exceeded'); return n
    def equal(x,y,label):
        a=x.contiguous().view(torch.uint8).flatten(); b=y.contiguous().view(torch.uint8).flatten()
        require(a.numel()==b.numel(),label+' size')
        for start in range(0,a.numel(),4*1024**2):
            aa,bb=a[start:start+4*1024**2],b[start:start+4*1024**2]; same=aa==bb
            if not bool(same.all()):
                i=int(torch.nonzero(~same,as_tuple=False)[0]); raise AssertionError(f'{label}: byte {start+i} actual={int(aa[i])} expected={int(bb[i])}')
    def tensor_sha(t):
        h=hashlib.sha256(); b=t.contiguous().view(torch.uint8).flatten()
        for start in range(0,b.numel(),4*1024**2): h.update(memoryview(b[start:start+4*1024**2].cpu().numpy()))
        return h.hexdigest()
    def fixture(rows):
        torch.cuda.reset_peak_memory_stats(); start=16384; end=start+rows; nb=(end+15)//16; pb=nb+3
        raw={}; data={}
        def alloc(name,shape,dtype):
            count=1
            for n in shape: count*=n
            size=count*torch.tensor([],dtype=dtype).element_size()
            r=torch.full((size+512,),0xa5,dtype=torch.uint8,device=dev)
            raw[name]=r; data[name]=r[256:-256].view(dtype).reshape(shape); return data[name]
        q=alloc('q',(rows,32,512),torch.bfloat16); q.normal_()
        kv=alloc('kv',(pb,16,512),torch.bfloat16); kv.normal_()
        table=alloc('table',(nb,),torch.int32); table.copy_(torch.arange(nb+1,1,-1,device=dev,dtype=torch.int32))
        selected=alloc('selected',(rows,2051),torch.int32); selected.fill_(-1)
        rr=torch.arange(rows,device=dev,dtype=torch.int32)
        selected[:,:2048]=((rr[:,None]*61+torch.arange(2048,device=dev,dtype=torch.int32)[None,:]*17+63)%start)
        length=start+rr+1; counts=(length%4).contiguous()
        for j in range(3): selected[:,2048+j]=torch.where(j<counts,length-counts+j,-1)
        qpad=alloc('qpad',(rows,32,576),torch.bfloat16)
        packed=alloc('packed_kv',((end+63)//64,64,656),torch.uint8)
        regions,nmeta=layout(rows); metadata=alloc('metadata',(nmeta,),torch.uint8)
        m={name:metadata[off:off+size].view(torch.float32 if 'lse' in name else torch.int32).reshape(shape) for name,(off,size,shape) in regions.items()}
        mo=alloc('main_out',q.shape,torch.bfloat16); to=alloc('tail_out',q.shape,torch.bfloat16); out=alloc('out',q.shape,torch.bfloat16)
        stream=torch.cuda.current_stream().cuda_stream
        args=Args(**{n:data[n].data_ptr() for n in list(data)},metadata_bytes=nmeta,stream=stream,rows=rows,seq_start=start,physical_blocks=pb,block_table_count=nb)
        hashes={n:tensor_sha(data[n]) for n in ['q','kv','selected','table']}
        if rows==2048:
            require(uni.atlas_glm_sparse_native_run(None)==-1,'null args rejection')
            tests=[(n,0,-1) for n in data]+[('rows',2047,-2),('rows',4101,-2),('seq_start',2047,-2),('seq_start',32768,-2),('physical_blocks',0,-2),('block_table_count',nb-1,-2),('metadata_bytes',nmeta-1,-3),('qpad',qpad.data_ptr()+2,-4),('out',q.data_ptr(),-4),('q',2**64-2,-4)]
            for name,value,code in tests:
                bad=Args.from_buffer_copy(args);setattr(bad,name,value)
                require(uni.atlas_glm_sparse_native_run(C.byref(bad))==code,'rejection '+name)
            for name in data:
                if name not in hashes: require(bool((raw[name]==0xa5).all()),'rejection wrote '+name)
            for n,h in hashes.items():require(tensor_sha(data[n])==h,'rejection changed '+n)
            report(stage='rejects',passed=True,cases=len(tests)+1)
        def unified(): require(uni.atlas_glm_sparse_native_run(C.byref(args))==0,'unified dispatch')
        unified();torch.cuda.synchronize()
        # Independent split/reference construction, never reading native metadata.
        mi=selected[:,:2048].contiguous(); ti=torch.full_like(mi,-1)
        ti[:,:3]=selected[:,2048:];ti[:,0]=torch.where(counts>0,ti[:,0],0)
        ml=torch.full((rows,),2048,dtype=torch.int32,device=dev);tl=counts.clamp_min(1)
        for name,value in [('main_ids',mi),('tail_ids',ti),('main_lengths',ml),('tail_lengths',tl),('tail_counts',counts)]: equal(m[name],value,'metadata '+name)
        pq=torch.zeros_like(qpad);pq[...,:512]=q
        logical=kv.index_select(0,table.long()).reshape(-1,512)[:end].float().reshape(end,4,128)
        scales=logical.abs().amax(-1).clamp_min(1e-4)/448.0
        quant=(logical/scales[...,None]).clamp(-448,448).to(torch.float8_e4m3fn)
        pk=torch.zeros_like(packed);flat=pk.view(-1,656)
        flat[:end,:512]=quant.contiguous().view(torch.uint8).reshape(end,512)
        flat[:end,512:528]=scales.contiguous().view(torch.uint8).reshape(end,16)
        flat[end:,512:528]=torch.ones((flat.shape[0]-end,4),device=dev).view(torch.uint8)
        equal(qpad,pq,'prepared Q');equal(packed,pk,'packed KV')
        del logical,scales,quant
        ref=torch.full_like(out,float('nan')); rl=torch.full((rows,32),float('nan'),device=dev)
        l0=torch.full_like(rl,float('nan'));l1=torch.full_like(rl,float('nan'))
        op(pq,pk,mi,mo,l0,.0625,2,ml,None,None,None,None)
        op(pq,pk,ti,to,l1,.0625,2,tl,None,None,None,None)
        require(merge.atlas_sparse_merge(mo.data_ptr(),to.data_ptr(),l0.data_ptr(),l1.data_ptr(),counts.data_ptr(),ref.data_ptr(),rl.data_ptr(),rows,32,stream)==0,'reference merge')
        torch.cuda.synchronize()
        require(bool(torch.isfinite(out).all()) and bool(torch.isfinite(m['final_lse']).all()),'unwritten/nonfinite unified')
        equal(out,ref,'unified output');equal(m['final_lse'],rl,'unified LSE')
        aout=torch.full_like(out,float('nan'))
        def baseline():
            require(atlas.atlas_sparse_run(q.data_ptr(),kv.data_ptr(),selected.data_ptr(),aout.data_ptr(),table.data_ptr(),rows,32,2051,16,.0625,stream)==0,'Atlas dispatch')
        baseline();torch.cuda.synchronize();require(bool(torch.isfinite(aout).all()),'Atlas nonfinite output')
        def checks():
            for n,h in hashes.items():require(tensor_sha(data[n])==h,'input changed '+n)
            for n,r in raw.items():require(bool((r[:256]==0xa5).all()) and bool((r[-256:]==0xa5).all()),'guard '+n)
            offset=0
            for off,size,_ in regions.values():
                require(bool((metadata[offset:off]==0xa5).all()),'metadata alignment padding');offset=off+size
        checks()
        report(stage='correctness',passed=True,rows=rows,seq_start=start,seq_end=end,output_bytes=out.numel()*2,lse_bytes=rl.numel()*4,peak_reserved_bytes=mem())
        if rows!=4096:return None
        del pq,pk,mi,ti,ml,tl,l0,l1
        functions={'atlas':baseline,'unified_inclusive':unified}
        for _ in range(3):
            for fn in functions.values():fn();torch.cuda.synchronize()
        timings={name:[] for name in functions}
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(18)]
        for pair in range(9):
            names=list(functions);names=names[pair%2:]+names[:pair%2]
            for j,name in enumerate(names):
                begin,finish=events[pair*2+j];begin.record();functions[name]();finish.record();finish.synchronize()
                timings[name].append(begin.elapsed_time(finish))
        equal(out,ref,'timed output');equal(m['final_lse'],rl,'timed LSE');checks()
        median={k:statistics.median(v) for k,v in timings.items()};gain=median['atlas']/median['unified_inclusive']
        report(stage='timing',rows=rows,warmups=3,pairs=9,samples_ms=timings,median_ms=median,inclusive_speedup=gain,minimum_speedup=1.3,candidate_pass=gain>=1.3,peak_reserved_bytes=mem())
        return gain
    for rows in [2048,3515,4100,4096]:
        gain=fixture(rows);gc.collect();torch.cuda.empty_cache()
    if gain<1.3:report(stage='rejected',reason='inclusive speedup below 1.3',inclusive_speedup=gain);return 3
    report(stage='complete',passed=True,fixtures=4,inclusive_speedup=gain);return 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unified',default='/eval/libatlas_glm_sparse_native.so')
    p.add_argument('--atlas',default='/eval/libatlas_sparse.so')
    p.add_argument('--native',default='/native/sparse_mla_sm120.so')
    p.add_argument('--merge',default='/eval/libatlas_sparse_merge.so')
    try:raise SystemExit(main(p.parse_args()))
    except Exception as e:report(stage='failure',passed=False,error=str(e));raise SystemExit(1)
