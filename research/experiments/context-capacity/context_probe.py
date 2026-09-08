#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Original synthetic context-capacity and tool-continuation probe.
import argparse, json, os, time, hashlib, urllib.request, urllib.error, pathlib, subprocess
ap = argparse.ArgumentParser()
ap.add_argument('--tokens', type=int, required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--tool-choice', choices=['named', 'auto'], default='named')
ap.add_argument('--replay-request')
a = ap.parse_args()
pid = subprocess.check_output(['systemctl', '--user', 'show', 'lloom.service', '--property=MainPID', '--value'], text=True).strip()
env = dict((x.split(b'=', 1) for x in pathlib.Path('/proc', pid, 'environ').read_bytes().split(b'\x00') if b'=' in x))
headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + env[b'LLOOM_API_KEY'].decode()}
base = 'http://127.0.0.1:8100'
backend = 'http://127.0.0.1:8890'
model = 'sparkglm-nvfp4'

def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=7200))

def metrics():
    try:
        return urllib.request.urlopen(backend + '/metrics', timeout=20).read().decode()
    except Exception as e:
        return '# metrics unavailable: ' + str(e) + '\n'
kwargs = {'thinking': False, 'enable_thinking': False}
tools = [{'type': 'function', 'function': {'name': 'record_code', 'description': 'Record the exact requested archive code.', 'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code'], 'additionalProperties': False}}}]
salt = os.urandom(8).hex()
codes = ['RAVEN-48271', 'MAPLE-69324', 'ORBIT-81539']
system = 'Synthetic archive test ' + salt + '. Follow the final user instruction. Use record_code to report the requested archive code exactly.'
lines = [f'Record {i:07d}: district {i % 97}, inventory {i * 7919 % 100003}, routine inspection complete.\n' for i in range(a.tokens // 12 + 2000)]
text = ''.join(lines)

def messages(n):
    doc = text[:n]
    p = len(doc) // 2
    doc = 'Archive START code: ' + codes[0] + '.\n' + doc[:p] + '\nArchive MIDDLE code: ' + codes[1] + '.\n' + doc[p:] + '\nArchive END code: ' + codes[2] + '.'
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': doc + '\nCall record_code with the START code.'}]

def count(m):
    return post(backend + '/tokenize', {'model': model, 'messages': m, 'tools': tools, 'add_generation_prompt': True, 'chat_template_kwargs': kwargs})['count']
if a.replay_request:
    replay = pathlib.Path(a.replay_request)
    prior = json.loads(replay.with_name(replay.name.replace('-request.json', '.json')).read_text())
    m = json.loads(replay.read_text())['messages']
    actual = prior['tokenized_prompt_tokens']
    lo = prior['document_prefix_characters']
    salt = prior['salt']
    assert hashlib.sha256(json.dumps(m).encode()).hexdigest() == prior['prompt_sha256']
    assert actual == a.tokens
else:
    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(messages(mid)) <= a.tokens:
            lo = mid
        else:
            hi = mid - 1
    m = messages(lo)
    actual = count(m)
out = {'requested_prompt_tokens': a.tokens, 'tokenized_prompt_tokens': actual, 'prompt_sha256': hashlib.sha256(json.dumps(m).encode()).hexdigest(), 'turns': [], 'started_at': time.time(), 'tool_choice': a.tool_choice, 'replayed_tokenization': bool(a.replay_request), 'salt': salt, 'document_prefix_characters': lo}
root = pathlib.Path(a.output)
root.parent.mkdir(parents=True, exist_ok=True)

def save():
    root.write_text(json.dumps(out, indent=2) + '\n')
save()
print(json.dumps({'stage': 'prompt-ready', 'tokens': actual}), flush=True)
for i in range(3):
    before = metrics()
    t = time.monotonic()
    first = None
    content = ''
    calls = {}
    usage = None
    finish = None
    err = None
    payload = {'model': model, 'messages': m, 'tools': tools, 'tool_choice': 'auto' if a.tool_choice == 'auto' else {'type': 'function', 'function': {'name': 'record_code'}}, 'temperature': 0, 'max_tokens': 128, 'chat_template_kwargs': kwargs, 'stream': True, 'stream_options': {'include_usage': True}}
    if i == 0:
        root.with_name(root.stem + '-request.json').write_text(json.dumps(payload))
    try:
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=7200) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    break
                e = json.loads(raw)
                if e.get('error'):
                    raise RuntimeError(str(e['error']))
                if e.get('usage'):
                    usage = e['usage']
                for choice in e.get('choices', []):
                    d = choice.get('delta', {})
                    finish = choice.get('finish_reason') or finish
                    if (d.get('content') or d.get('tool_calls')) and first is None:
                        first = time.monotonic() - t
                    content += d.get('content') or ''
                    for call in d.get('tool_calls', []):
                        c = calls.setdefault(call.get('index', 0), {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        if call.get('id'):
                            c['id'] = call['id']
                        for k in ['name', 'arguments']:
                            c['function'][k] += call.get('function', {}).get(k) or ''
    except urllib.error.HTTPError as e:
        err = f'HTTP {e.code}: {e.read().decode()}'
    except Exception as e:
        err = str(e)
    after = metrics()
    root.with_name(root.stem + f'-turn{i}-before.prom').write_text(before)
    root.with_name(root.stem + f'-turn{i}-after.prom').write_text(after)
    tc = list(calls.values())
    ok = False
    try:
        ok = len(tc) == 1 and tc[0]['function']['name'] == 'record_code' and (json.loads(tc[0]['function']['arguments']) == {'code': codes[i]})
    except Exception:
        pass
    result = {'turn': i, 'ttft_s': first, 'wall_s': time.monotonic() - t, 'usage': usage, 'finish_reason': finish, 'content': content, 'tool_calls': tc, 'correct': ok, 'finish_reason_is_tool_calls': finish == 'tool_calls', 'error': err}
    out['turns'].append(result)
    save()
    print(json.dumps(result), flush=True)
    if err or not tc:
        break
    m.append({'role': 'assistant', 'content': content or None, 'tool_calls': tc})
    m.append({'role': 'tool', 'tool_call_id': tc[0]['id'], 'content': 'Recorded successfully.'})
    if i < 2:
        m.append({'role': 'user', 'content': 'Call record_code with the ' + ['MIDDLE', 'END'][i] + ' code.'})
out['completed_at'] = time.time()
save()
