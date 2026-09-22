# SPDX-License-Identifier: Apache-2.0
"""Four staggered continuations after GPU eviction, one shared 200K prefix."""
import concurrent.futures
import json
from pathlib import Path
import time
import cache_bench as bench

saved=json.loads(Path('/tmp/full-cache-result.json').read_text())
ids=saved['prompt_ids']
assert len(ids)==200000
assert bench.post('/reset_prefix_cache')=={'success':True}
before=bench.metrics()
def one(index):
    time.sleep(index*0.5)
    return bench.request(ids,'staggered-'+str(index))
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(one,range(4)))
report={'workload':'four staggered restores/continuations of one shared 200K prefix','stagger_seconds':0.5,'seconds':time.monotonic()-start,'before':before,'after':bench.metrics(),'requests':results}
Path('/tmp/concurrent-cache-result.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(r['output_sha256']==saved['hot']['output_sha256'] for r in results),'Concurrent continuation differs'
print('All four staggered continuations matched the single hot-cache reference.',flush=True)
