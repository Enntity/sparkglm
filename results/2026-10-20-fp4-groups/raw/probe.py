# SPDX-License-Identifier: AGPL-3.0-only
"""Six timed requests and six existing quality cases per precision policy."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time
import bench
import grader

ROOT = Path(__file__).resolve().parent
arm = sys.argv[1]
prompts, prompt_sha = bench.load_prompts(ROOT / 'prompts.json')
client = bench.Client('http://127.0.0.1:8893', '', 300)
result = {'arm': arm, 'started_at': time.time(), 'decode': {}, 'requests': {}, 'quality': []}

def save():
    (ROOT / (arm+'-result.json')).write_text(json.dumps(result, indent=2)+'\n')

warm = bench.build_chat_payload('glm-5.3-flash-atlas', prompts['system'],
    'Briefly explain how a lighthouse helps ships navigate.', 64, 20260905,
    {'chat_template_kwargs': {'reasoning_effort': 'low'}})
result['warmup'] = client.stream('/v1/chat/completions', warm)
for name, source in [('prose', 'shared'), ('code', 'rc4'), ('structured', 'shared')]:
    baseline = json.loads((ROOT / (source+'-baseline.json')).read_text())
    assert baseline['protocol']['prompts_sha256'] == prompt_sha
    settings, cid = baseline['settings'], baseline['run']['comparison_id']
    result['requests'][name] = [bench.build_chat_payload(baseline['run']['model'],
        prompts['system'], f"Request nonce: {bench.nonce(cid, 'decode', name, i+1)}\n\n{prompts['workloads'][name]}",
        settings['decode_tokens'], settings['seed'], settings['extra_body']) for i in range(2)]
    selected = dict(prompts, workloads={name: prompts['workloads'][name]})
    result['decode'].update(bench.run_decode(client, baseline['run']['model'], selected,
        2, settings['decode_tokens'], settings['seed'], settings['extra_body'], cid))
    save()
    print(arm, name, result['decode'][name]['decode_tokens_per_second'], flush=True)

cases = json.loads((ROOT/'cases.json').read_text())
def quality(case):
    payload = {'model':'glm-5.3-flash-atlas','max_tokens':8192,'temperature':0,
        'top_k':1,'top_p':1,'min_p':0,'repetition_penalty':1,'disable_mtp':False,
        'chat_template_kwargs':{'reasoning_effort':'low'},
        'messages':[{'role':'user','content':case['prompt']}]}
    row = {'id':case['id'],'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
    started = time.monotonic()
    try:
        row['response'] = client.json('/v1/chat/completions',payload)
    except Exception as e:
        row['error'] = repr(e)
    row['elapsed_seconds'] = time.monotonic()-started
    row['grade'] = grader.grade(case,row)
    return row

with concurrent.futures.ThreadPoolExecutor(4) as pool:
    for row in pool.map(quality,cases[:4]):
        result['quality'].append(row); save(); print(arm,row['id'],row['grade']['status'],flush=True)
for case in cases[4:]:
    row=quality(case);result['quality'].append(row);save();print(arm,row['id'],row['grade']['status'],flush=True)
result['finished_at']=time.time()
result['quality_passed']=all(r['grade']['status']=='pass' for r in result['quality'])
result['speed_finished']=all(x['completion_gate']['passed']==2 for x in result['decode'].values())
save()
