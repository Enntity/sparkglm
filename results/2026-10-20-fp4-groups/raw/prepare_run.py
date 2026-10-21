# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
import json,shutil
r=Path(__file__).resolve().parent;old=r.parent/'glm-mla-fusion-20261020';product=r.parent/'glm-native-mtp-20261020/product'
for n in ('bench.py','prompts.json','rc4-baseline.json','shared-baseline.json'):shutil.copy2(old/n,r/n)
shutil.copy2(product/'install/start-node.sh',r/'start-node.sh')
base=json.loads((product/'install/profiles/4x1m.json').read_text())
for arm,shared,dense in [('control','1','1'),('shared','0','1'),('dense','1','0'),('both','0','0')]:
 p=json.loads(json.dumps(base));p['environment'].update(ATLAS_GLM_FP4L_SHARED_FP8=shared,ATLAS_GLM_FP4L_DENSE_MXFP8=dense)
 (r/(arm+'.json')).write_text(json.dumps(p,indent=2)+'\n')
s=(r.parent/'glm-tp2-modes-20261020/grade.py').read_text().split('summary = {')[0]
(r/'grader.py').write_text(s)
s=(old/'run.py').read_text();s=s[:s.index("current_arm='off'")]
s=s.replace('glm-mla-fusion-20261020','glm-fp4-groups-20261020').replace('codex-glm-mla-fusion','codex-glm-fp4-groups').replace('atlas-sparkglm-mla-fuse','atlas-sparkglm-fp4-groups')
s=s.replace("['start-node.sh', 'probe.py', 'bench.py', 'prompts.json', 'rc4-baseline.json', 'shared-baseline.json','parity.py','off.json','on.json']","['start-node.sh','probe.py','bench.py','prompts.json','rc4-baseline.json','shared-baseline.json','control.json','shared.json','dense.json','both.json','cases.json','grader.py','check_code.py']")
(r/'run.py').write_text(s)
