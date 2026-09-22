# SPDX-License-Identifier: Apache-2.0
"""One synthetic restart-persistence proof; no cache-reset endpoint required."""
import json
from pathlib import Path
import cache_bench as b
b.BASE='http://127.0.0.1:8892'
b.MODEL='sparkglm-nvfp4-nvidia'
saved=json.loads(Path('/tmp/nvme-proof/full-cache-result.json').read_text())
result=b.request(saved['prompt_ids'],'restart_restore')
Path('/tmp/nvme-restart-result.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['usage']['prompt_tokens_details']['cached_tokens']>=180000
assert result['output_sha256']==saved['hot']['output_sha256']
assert result['ttft_seconds']<saved['cold']['ttft_seconds']*.8
assert result['disk_io_delta']['read_bytes']>0
print('Restart reused SSD prefix with byte-identical continuation and physical reads.')
