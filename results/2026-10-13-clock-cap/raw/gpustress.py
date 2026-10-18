#!/usr/bin/env python3
"""Sustained full-power GPU load for the 02 power-off investigation.

bf16 8192^3 matmuls back to back (about 3 GB of device memory). Every 10 s it prints
achieved TFLOPS and GPU temperature. It stops after DURATION_S, or at once if the GPU
reaches MAX_C. Usage: gpustress.py [duration_s] [max_c]
"""
import subprocess, sys, time
import torch

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 600
max_c = int(sys.argv[2]) if len(sys.argv) > 2 else 95
n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
c = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)

def gpu_temp():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [x.strip() for x in out.split(",")]

start = last = time.time()
iters = 0
while True:
    for _ in range(20):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    iters += 20
    now = time.time()
    if now - last >= 10:
        temp, power, sm = gpu_temp()
        tflops = iters * 2 * n**3 / (now - last) / 1e12
        print(f"{time.strftime('%T')} t={now-start:5.0f}s {tflops:6.1f} TFLOPS gpu={temp}C {power}W sm={sm}MHz", flush=True)
        iters, last = 0, now
        if int(temp) >= max_c:
            print(f"STOP: GPU reached {temp}C", flush=True)
            break
    if now - start >= duration:
        print("DONE: full duration", flush=True)
        break
