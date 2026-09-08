#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source-locked opt-in adapter insertion; never edits the frozen manifest."""
import hashlib
import json
from pathlib import Path
import sys

target=Path(sys.argv[1])
source=target.read_text()
if hashlib.sha256(target.read_bytes()).hexdigest() != '7eb1aa9bd0a61f62a0bfa6f762890db3a33995586596ceb7edbee4870ca07c2f':
    raise SystemExit('reference EXL3 source hash mismatch')
anchor='''        expert_offsets = torch.cumsum(expert_count, dim=0).sub(expert_count)
        scratch = _grouped_prefill_scratch('''
insert='''        if (os.environ.get("SPARKGLM_EXL3_E3", "0") == "1"
                and xh.shape[0] >= (2048 if cap == 32 else 4096)):
            from sparkglm_e3 import apply as apply_e3, eligible as eligible_e3
            if eligible_e3(xh, token_sorted.numel(), int(temps[2].shape[2])):
                apply_e3(xh, out, counts, token_sorted, weight_sorted,
                         ptrs, int(temps[2].shape[2]), cap, limit)
                _record_exl3_fat_tier(layer, "kernel", "sparkglm_e3_large_prefill")
                return out
'''
if source.count(anchor)!=1: raise SystemExit('grouped dispatch anchor drift')
before=hashlib.sha256(target.read_bytes()).hexdigest()
source=source.replace(anchor,insert+anchor)
compile(source,str(target),'exec')
target.write_text(source)
print(json.dumps({'path':str(target),'reference_sha256':before,
                  'candidate_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
