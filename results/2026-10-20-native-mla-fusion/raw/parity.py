# SPDX-License-Identifier: AGPL-3.0-only
"""Focused normal-serving lifecycle checks and two small matched decode pairs."""
import concurrent.futures as cf
import json
from pathlib import Path
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'comparison.json'
record={'warmups':[], 'correctness':[], 'concurrency':[], 'long_context':[], 'timings':[]}
BASE={'model':'glm-5.3-flash-atlas','temperature':0,'top_k':1,'top_p':1,'min_p':0,
      'repetition_penalty':1,'chat_template_kwargs':{'enable_thinking':False,'reasoning_effort':'low'},
      'stream':True,'stream_options':{'include_usage':True},'return_token_ids':True}
URL='http://127.0.0.1:8893'
def save():
 OUT.write_text(json.dumps(record,indent=2)+'\n')
 print('Checkpoint:',{k:len(v) if isinstance(v,list) else ('saved' if isinstance(v,dict) else v) for k,v in record.items()},flush=True)
def accepted(row): return row['usage'].get('completion_tokens_details',{}).get('accepted_prediction_tokens',0)
def cached(row): return row['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)
def request(prompt,budget,serial=False,extra=None,barrier=None,cancel=False):
 body=dict(BASE,messages=[{'role':'user','content':prompt}],max_tokens=budget,disable_mtp=serial)
 body.update(extra or {})
 req=urllib.request.Request(URL+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 if barrier: barrier.wait(timeout=20)
 start=time.monotonic(); first=None
 row={'request':body,'events':[],'token_ids':[],'content':'','reasoning_content':'','started':start}
 with urllib.request.urlopen(req,timeout=360) as response:
  for raw in response:
   if not raw.startswith(b'data:'): continue
   data=raw[5:].strip()
   if data==b'[DONE]': break
   event=json.loads(data); row['events'].append(event)
   if event.get('error'): raise AssertionError(event)
   if event.get('usage'): row['usage']=event['usage']
   for choice in event.get('choices',[]):
    delta=choice.get('delta') or {}
    ids=choice.get('token_ids') or delta.get('token_ids') or []
    row['token_ids'].extend(ids)
    for key in ('content','reasoning_content'): row[key]+=delta.get(key) or ''
    if (ids or delta.get('content')) and first is None: first=time.monotonic()
    if choice.get('finish_reason'): row['finish_reason']=choice['finish_reason']
   if cancel and row['token_ids']:
    row['cancelled_after_tokens']=len(row['token_ids']);break
 row['ended']=time.monotonic();row['elapsed_seconds']=row['ended']-start
 row['ttft_seconds']=None if first is None else first-start
 if cancel: return row
 assert row['token_ids'], 'Missing token IDs'
 assert row.get('finish_reason') in ('stop','length','tool_calls'), row
 assert row.get('usage'), 'Missing final usage'
 if serial: assert accepted(row)==0,'Serial control used draft tokens'
 return row

import sys
arm=sys.argv[1]
checks=[
 ('arithmetic','What is 17 + 25? Reply with only the integer.',32),
 ('json','Sort 8, 2, 5 in ascending order. Reply only with a JSON array.',32),
 ('prose','Write a calm, detailed essay about how lighthouses were staffed and maintained in the 1800s.',128),
 ('code','Write a Python function merge_intervals(intervals) that merges overlapping closed intervals without modifying its input. Include examples.',128),
 ('structured','Return only a JSON object with three planets, each containing name, orbital_period_days, and moons. Use Mercury, Venus, and Earth.',128),
]
rows={}
for name,prompt,budget in checks:
 rows[name]=request(prompt,budget)
 (ROOT/('parity-'+arm+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
 print(arm,name,len(rows[name]['token_ids']),rows[name].get('finish_reason'),flush=True)
assert rows['arithmetic']['content'].strip()=='42'
assert json.loads(rows['json']['content'])==[2,5,8]
