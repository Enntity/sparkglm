# SPDX-License-Identifier: AGPL-3.0-only
import json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
deadline=time.monotonic()+600
while not (r/'build.log').exists():
    assert time.monotonic()<deadline,'builder readiness timeout'
    time.sleep(2)
command=['ssh','-o','BatchMode=yes','ennspark01','docker run --rm --gpus all --network none --memory 8g -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/aarch64-linux-gnu -e ATLAS_PDL=1 --entrypoint /bin/bash localhost/atlas-mla-fuse-builder:20261020 -lc '+"'cd /build && cargo test --locked --release -j 4 -p spark-model --lib glm_mla_fuse -- --ignored --nocapture --test-threads=1'"]
with (r/'gpu-tests.log').open('wb') as log:
    p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=480)
(r/'gpu-tests.json').write_text(json.dumps({'returncode':p.returncode,'pdl':'1','command':command})+'\n')
assert p.returncode==0,'GPU byte parity failed; see gpu-tests.log'
print('GPU byte parity passed',flush=True)
