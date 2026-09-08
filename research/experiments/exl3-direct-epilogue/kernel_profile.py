#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: profiling perturbs timing; never use this as an A/B receipt."""
import json
import torch
from bench import run_case

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
    run_case([129, 192, 256, 383, 512, 640, 768, 1024], 20260906, 2, 5, 2)
print(json.dumps([
    {'kernel': event.key, 'count': event.count,
     'total_device_us': event.device_time_total,
     'mean_device_us': event.device_time_total / event.count}
    for event in profiler.key_averages() if 'exl3_fat_grouped_' in event.key
], indent=2))
