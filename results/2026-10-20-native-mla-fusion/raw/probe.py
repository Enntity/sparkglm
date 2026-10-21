# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import time
import bench

ROOT = Path(__file__).resolve().parent
prompts, prompt_sha = bench.load_prompts(ROOT / 'prompts.json')
client = bench.Client('http://127.0.0.1:8893', '', 180)
result = {'started_at': time.time(), 'decode': {}, 'requests': {}, 'baselines': {}}
# A short unmeasured warmup avoids timing first-use kernel compilation.
warm = bench.build_chat_payload('glm-5.3-flash-atlas', prompts['system'],
    'Briefly explain how a lighthouse helps ships navigate.', 64, 20260905,
    {'chat_template_kwargs': {'reasoning_effort': 'low'}})
result['warmup'] = client.stream('/v1/chat/completions', warm)
for name, source in [('prose', 'shared'), ('code', 'rc4'), ('structured', 'shared')]:
    baseline = json.loads((ROOT / (source + '-baseline.json')).read_text())
    assert baseline['protocol']['prompts_sha256'] == prompt_sha
    settings = baseline['settings']
    cid = baseline['run']['comparison_id']
    result['baselines'][name] = {'source': source, 'comparison_id': cid,
        'median': baseline['decode'][name]['decode_tokens_per_second']['median']}
    result['requests'][name] = [bench.build_chat_payload(baseline['run']['model'],
        prompts['system'], f"Request nonce: {bench.nonce(cid, 'decode', name, i+1)}\n\n{prompts['workloads'][name]}",
        settings['decode_tokens'], settings['seed'], settings['extra_body']) for i in range(3)]
    selected = dict(prompts, workloads={name: prompts['workloads'][name]})
    result['decode'].update(bench.run_decode(client, baseline['run']['model'], selected,
        3, settings['decode_tokens'], settings['seed'], settings['extra_body'], cid))
    (ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
result['finished_at'] = time.time()
result['prompts_sha256'] = prompt_sha
result['harness_sha256'] = hashlib.sha256((ROOT / 'bench.py').read_bytes()).hexdigest()
(ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
assert all(x['completion_gate']['passed'] == 3 for x in result['decode'].values())
