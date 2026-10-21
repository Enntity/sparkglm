# SPDX-License-Identifier: AGPL-3.0-only
"""Same existing uncapped-clock workload and output budget; bounded repetitions."""
import argparse,hashlib,json,time
from pathlib import Path
import bench
p=argparse.ArgumentParser();p.add_argument('--runs',type=int,default=2);p.add_argument('--serial',action='store_true');o=p.parse_args()
ROOT=Path(__file__).resolve().parent;OUT=Path.cwd()/('speed-serial.json' if o.serial else 'speed.json')
prompts,prompt_sha=bench.load_prompts(ROOT/'prompts.json');client=bench.Client('http://127.0.0.1:8893','',300)
result={'started':time.time(),'decode':{},'requests':{},'serial':o.serial,'runs':o.runs,'prompts_sha256':prompt_sha}
def save():OUT.write_text(json.dumps(result,indent=2)+'\n')
warm=bench.build_chat_payload('glm-5.3-flash-atlas',prompts['system'],'Briefly explain how a lighthouse helps ships navigate.',64,20260905,{'chat_template_kwargs':{'reasoning_effort':'low'},'disable_mtp':o.serial})
result['warmup']=client.stream('/v1/chat/completions',warm)
for name,source in [('prose','shared'),('code','rc4'),('structured','shared')]:
 baseline=json.loads((ROOT/(source+'-baseline.json')).read_text());assert baseline['protocol']['prompts_sha256']==prompt_sha
 settings,cid=baseline['settings'],baseline['run']['comparison_id'];extra=dict(settings['extra_body'],disable_mtp=o.serial)
 result['requests'][name]=[bench.build_chat_payload(baseline['run']['model'],prompts['system'],f"Request nonce: {bench.nonce(cid,'decode',name,i+1)}\n\n{prompts['workloads'][name]}",settings['decode_tokens'],settings['seed'],extra) for i in range(o.runs)]
 selected=dict(prompts,workloads={name:prompts['workloads'][name]})
 result['decode'].update(bench.run_decode(client,baseline['run']['model'],selected,o.runs,settings['decode_tokens'],settings['seed'],extra,cid));save()
 assert result['decode'][name]['completion_gate']['passed']==o.runs
 print(name,result['decode'][name]['decode_tokens_per_second'],flush=True)
result['ended']=time.time();result['passed']=True;save()
