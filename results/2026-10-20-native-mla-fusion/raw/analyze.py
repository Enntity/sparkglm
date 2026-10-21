# SPDX-License-Identifier: AGPL-3.0-only
import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
new=json.loads((r/'result.json').read_text())
old=json.loads((r.parent/'glm-spark-nvfp4-quick-20261020/result.json').read_text())
report={}
for name in ['prose','code','structured']:
    a=old['decode'][name];b=new['decode'][name]
    previous=a['decode_tokens_per_second']['median'];current=b['decode_tokens_per_second']['median'];nvfp4=new['baselines'][name]['median']
    matches=[]
    for i,(x,y) in enumerate(zip(a['runs'],b['runs']),1):
        fields=['prompt_tokens','completion_tokens','output_sha256','reasoning_sha256','finish_reason']
        matches.append({'run':i,'exact':all(x[k]==y[k] for k in fields),'different_fields':[k for k in fields if x[k]!=y[k]]})
    report[name]={'before':previous,'after':current,'nvfp4':nvfp4,'gain_pct':100*(current/previous-1),'vs_nvfp4_pct':100*(current/nvfp4-1),'repetitions':[x['decode_tokens_per_second'] for x in b['runs']],'output_matches_previous':matches}
(r/'comparison-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
