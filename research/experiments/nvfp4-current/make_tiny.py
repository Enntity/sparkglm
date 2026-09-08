#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build metadata for a deterministic, weightless NVFP4 integration fixture."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[3]
expected=json.loads((Path(__file__).parent/'candidate.json').read_text())['target']['config_sha256']
data=Path(a.config).read_bytes()
if hashlib.sha256(data).hexdigest()!=expected:raise SystemExit('pinned checkpoint config hash mismatch')
quant=json.loads(data)['quantization_config']
quant['config_groups']={'group_0':quant['config_groups']['group_0']}
quant['config_groups']['group_0']['targets']=[r're:.*\.mlp\.experts\..*(gate|up|down)_proj$']
quant['ignore']=[]
output=Path(a.output)
if (output/'config.json').exists():
    old=json.loads((output/'config.json').read_text())
    if old.get('_sparkglm_fixture')!='tinyglm-nvfp4-v1':raise SystemExit('refusing to overwrite a non-fixture directory')
spec=importlib.util.spec_from_file_location('make_tinyglm',root/'scripts/make_tinyglm.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
    snapshot=m.build(Path(tmp),16,256,32768)
    c=json.loads((snapshot/'config.json').read_text())
    c['quantization_config']=quant;c['_sparkglm_fixture']='tinyglm-nvfp4-v1'
    (snapshot/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    (snapshot/'quantization_config.json').write_text(json.dumps(quant,indent=2)+'\n')
    shutil.copytree(snapshot,output,dirs_exist_ok=True)
print(output)
