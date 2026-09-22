# SPDX-License-Identifier: Apache-2.0
"""Synthetic cold/hot/SSD-restored probe against an isolated serving port."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

BASE='http://127.0.0.1:8893'
MODEL='sparkglm-nvme-test'

def post(path, body=None, timeout=600):
    data=json.dumps(body or {}).encode()
    with urllib.request.urlopen(urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json'}),timeout=timeout) as r:
        raw=r.read()
        return json.loads(raw) if raw else None

def metrics():
    with urllib.request.urlopen(BASE+'/metrics',timeout=20) as r:
        lines=r.read().decode().splitlines()
    return [s for s in lines if not s.startswith('#') and any(k in s for k in ('prefix_cache','num_preemptions','num_requests_running','num_requests_waiting','offload'))]

def io_counters():
    totals={'read_bytes':0,'write_bytes':0}
    for path in Path('/proc').glob('[0-9]*/io'):
        try:
            for line in path.read_text().splitlines():
                key,value=line.split(':',1)
                if key in totals:totals[key]+=int(value)
        except (OSError, ValueError):pass
    return totals


def request(ids, label, tiny=False):
    body={'model':MODEL,'prompt':ids,'stream':True,'stream_options':{'include_usage':True},'temperature':0,'seed':917,'max_tokens':32,'logprobs':1,'ignore_eos':tiny}
    disk_before=io_counters()
    start=time.monotonic();first=None;chunks=[];usage={};tokens=[]
    with urllib.request.urlopen(urllib.request.Request(BASE+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=900) as r:
        for line in r:
            if not line.startswith(b'data: '):continue
            line=line[6:].strip()
            if line==b'[DONE]':break
            d=json.loads(line)
            if d.get('usage'):usage=d['usage']
            for c in d.get('choices',[]):
                if c.get('text'):
                    if first is None:first=time.monotonic()-start
                    chunks.append(c['text'])
                tokens.extend((c.get('logprobs') or {}).get('tokens') or [])
    text=''.join(chunks)
    result={'label':label,'ttft_seconds':first,'wall_seconds':time.monotonic()-start,'usage':usage,'text':text,'tokens':tokens,'output_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':len(ids),'prompt_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    disk_after=io_counters()
    result['disk_io_delta']={k:disk_after[k]-disk_before[k] for k in disk_before}
    print(json.dumps(result),flush=True)
    assert first is not None and usage.get('completion_tokens',0)>0
    return result

def settle(cache):
    # Only synthetic experiment cache; do not inspect any entity content.
    previous=None;stable=0
    for _ in range(45):
        files=list(Path(cache).rglob('*.bin'))
        state=(len(files),sum(f.stat().st_size for f in files))
        if state==previous:stable+=1
        else:stable=0
        if stable>=5:return {'files':state[0],'bytes':state[1]}
        previous=state;time.sleep(1)
    raise RuntimeError('SSD writes did not quiesce')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tiny',action='store_true');ap.add_argument('--tokens',type=int,default=200000);ap.add_argument('--phase',choices=['all','restore','fault'],default='all');ap.add_argument('--out',required=True);ap.add_argument('--cache',required=True);a=ap.parse_args()
    out=Path(a.out)
    if a.phase!='all':
        saved=json.loads(out.read_text());ids=saved['prompt_ids']
        assert post('/reset_prefix_cache') == {'success':True}, 'GPU cache reset failed'
        label=a.phase+'_restore';saved[label]=request(ids,label,a.tiny);saved['after_'+label]=metrics()
        out.write_text(json.dumps(saved,indent=2)+'\n')
        assert saved[label]['output_sha256']==saved['hot']['output_sha256'], 'Continuation changed after '+label
        return
    if a.tiny:
        rng=random.Random(918);ids=[rng.randrange(3,250) for _ in range(a.tokens)]
    else:
        unit='\n'.join(f'Record {i:05d}: garden cedar amber cobalt orbit; quantity {i%97:02d}; verified archive entry.' for i in range(100))
        unit_tokens=post('/tokenize',{'model':MODEL,'prompt':unit})['count']
        n=int(a.tokens*1.08/unit_tokens*100)+100
        filler='\n'.join(f'Record {i:05d}: garden cedar amber cobalt orbit; quantity {i%97:02d}; verified archive entry.' for i in range(n))
        messages=[{'role':'user','content':'Read these synthetic archive records.\n'+filler+'\nThe final verification code is AZURE-7319. Repeat only that code.'}]
        tok=post('/tokenize',{'model':MODEL,'messages':messages,'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})
        ids=tok['tokens'];assert len(ids)>a.tokens;ids=ids[:a.tokens-128]+ids[-128:]
    report={'prompt_ids':ids,'before':metrics(),'cold':request(ids,'cold',a.tiny)}
    report['disk_after_cold']=settle(a.cache)
    report['hot']=request(ids,'hot',a.tiny)
    report['before_reset']=metrics()
    out.write_text(json.dumps(report,indent=2)+'\n')
    assert post('/reset_prefix_cache') == {'success':True}, 'GPU cache reset failed'
    report['after_reset']=metrics()
    report['restore']=request(ids,'restore',a.tiny)
    report['after']=metrics();report['disk_after_restore']=settle(a.cache)
    out.write_text(json.dumps(report,indent=2)+'\n')
    assert report['disk_after_cold']['files']>0,'No SSD cache files produced'
    assert report['restore']['output_sha256']==report['hot']['output_sha256'],'Restored continuation differs from hot-cache continuation'
    cold=report['cold']['ttft_seconds'];warm=report['restore']['ttft_seconds']
    if not a.tiny:
        assert warm<cold*.8, f'SSD restore not at least 20% faster: {warm} vs {cold}'

if __name__=='__main__':main()
