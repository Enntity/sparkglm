# SPDX-License-Identifier: Apache-2.0
import json, statistics, time
import torch
import vllm._C_stable_libtorch

torch.set_num_threads(1)
x = torch.ones((1, 4096), dtype=torch.bfloat16, device='cuda')
w = torch.ones(4096, dtype=torch.bfloat16, device='cuda')
out = torch.empty_like(x)
def operation():
    torch.ops._C.rms_norm(out, x, w, 1e-6)
for _ in range(50): operation()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph): operation()
results = {}
for name, fn in [('native_rmsnorm_launch', operation), ('graph_replay_control', graph.replay)]:
    samples = []
    for _ in range(15):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(1000): fn()
        end = time.perf_counter_ns()
        torch.cuda.synchronize()
        samples.append((end-start)/1e6)
    results[name] = {'host_us_per_call': statistics.median(samples), 'samples_us_per_call': samples}
print(json.dumps(results))
