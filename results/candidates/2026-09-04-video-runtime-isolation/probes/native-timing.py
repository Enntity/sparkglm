# SPDX-License-Identifier: Apache-2.0
import gc,json,statistics,time
import torch
import vllm._C_stable_libtorch

torch.set_num_threads(1)
torch.manual_seed(17)
if torch.cuda.mem_get_info()[0] < 2*1024**3:
    raise RuntimeError('Insufficient free CUDA memory for bounded probe')
results=[]
def timing(label,fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32): fn()
    for _ in range(3): graph.replay()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(11):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record();end.synchronize()
        samples.append(start.elapsed_time(end)*1000/32)
    results.append({'case':label,'median_us':statistics.median(samples),'samples_us':samples})
    del graph
for rows in (1,4,32,512,7168):
    x=torch.randn((rows,4096),device='cuda',dtype=torch.bfloat16)
    w=torch.ones(4096,device='cuda',dtype=torch.bfloat16);out=torch.empty_like(x)
    timing(f'rmsnorm-{rows}x4096',lambda:torch.ops._C.rms_norm(out,x,w,1e-6))
    del x,w,out
for rows in (1,4,32,512,7168):
    for cols in (16384,32768):
        x=torch.randn((rows,cols),device='cuda',dtype=torch.float16)
        end=torch.full((rows,),cols,device='cuda',dtype=torch.int32)
        start=torch.zeros_like(end);out=torch.empty((rows,512),device='cuda',dtype=torch.int32)
        if rows>=512:
            fn=lambda:torch.ops._C.top_k_per_row_prefill(x,start,end,out,rows,cols,1,512)
        else:
            fn=lambda:torch.ops._C.top_k_per_row_decode(x,1,end,out,rows,cols,1,512)
        timing(f'topk-fp16-{rows}x{cols}',fn)
        del x,start,end,out,fn
        gc.collect();torch.cuda.empty_cache()
print(json.dumps({'cases':results,'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name()},sort_keys=True))
