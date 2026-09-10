#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded synthetic serving semantics; not a general model quality benchmark."""
import argparse
import base64
import hashlib
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def red_png():
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    return base64.b64encode(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*64)*64))+chunk(b'IEND',b'')).decode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    destination=Path(a.output)
    if destination.exists():raise SystemExit('output already exists')
    report=dict(schema='sparkglm.bounded-semantics/v1',model=a.model,results=[],status='running',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def request(extra):
        payload=dict(model=a.model,temperature=0,max_tokens=512,chat_template_kwargs={'enable_thinking':False})
        payload.update(extra)
        req=urllib.request.Request(a.endpoint.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        start=time.monotonic()
        with urllib.request.urlopen(req,timeout=180) as r:d=json.load(r)
        return d,time.monotonic()-start,hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def run(name,messages,check,**extra):
        # Allow explicit overrides without duplicate keyword construction.
        payload=dict(messages=messages,**extra)
        started=time.monotonic()
        try:
            d,elapsed,signature=request(payload)
            message=d['choices'][0]['message']
            ok=bool(check(message))
            item=dict(name=name,pass_=ok,seconds=elapsed,request_sha256=signature,response=d)
        except Exception as exc:item=dict(name=name,pass_=False,seconds=time.monotonic()-started,error=str(exc))
        report['results'].append(item)
        destination.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in item.items() if k not in ('response','request_sha256')}),flush=True)
        return item
    user=lambda s:[{'role':'user','content':s}]
    text=lambda m:(m.get('content') or '').strip()
    run('plain',user('Reply with exactly SPARK_OK.'),lambda m:text(m)=='SPARK_OK')
    for x,y in [(17,19),(37,43),(123,47),(91,89)]:
        run(f'arithmetic_{x}_{y}',user(f'Compute {x} times {y}. Reply only with the integer.'),lambda m,n=x*y:text(m)==str(n))
    run('reasoning',user('Compute 37 times 43. Put the integer answer in your final response.'),lambda m:'1591' in text(m),chat_template_kwargs={'enable_thinking':True})
    schema={'type':'json_schema','json_schema':{'name':'answer','strict':True,'schema':{'type':'object','properties':{'city':{'type':'string'},'number':{'type':'integer'}},'required':['city','number'],'additionalProperties':False}}}
    run('structured',user('Return city Paris and number 17.'),lambda m:json.loads(text(m))=={'city':'Paris','number':17},response_format=schema)
    tools=[{'type':'function','function':{'name':'get_weather','description':'Read the weather for a city.','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False}}}]
    run('tools_parallel',user('Use get_weather for both Paris and Tokyo. Make both tool calls now.'),lambda m:sorted(json.loads(c['function']['arguments'])['city'] for c in m.get('tool_calls',[]) if c['function']['name']=='get_weather')==['Paris','Tokyo'],tools=tools,tool_choice='auto',parallel_tool_calls=True)
    calls=[{'id':'weather_p','type':'function','function':{'name':'get_weather','arguments':'{"city":"Paris"}'}},{'id':'weather_t','type':'function','function':{'name':'get_weather','arguments':'{"city":"Tokyo"}'}}]
    run('tool_results',[{'role':'user','content':'Which city is warmer? Reply only with the city.'},{'role':'assistant','content':None,'tool_calls':calls},{'role':'tool','tool_call_id':'weather_t','content':'{"city":"Tokyo","temperature_c":25}'},{'role':'tool','tool_call_id':'weather_p','content':'{"city":"Paris","temperature_c":12}'}],lambda m:text(m)=='Tokyo',tools=tools)
    filler='\n'.join(f'Reference row {i}: ordinary unchanged filler content.' for i in range(1900))
    for name,key in [('prefix_miss','ORCHID_7319'),('prefix_hit','ORCHID_7319'),('prefix_isolation','MAPLE_2806')]:
        run(name,user(filler+'\nThe exact audit key is '+key+'. Reply only with that key.'),lambda m,k=key:text(m)==k)
    run('stop',user('Output exactly this text: RED STOP BLUE'),lambda m:text(m)=='RED',stop=[' STOP'])
    run('vision_red',[{'role':'user','content':[{'type':'text','text':'What color is this square? Reply only with red, green, or blue.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+red_png()}}]}],lambda m:text(m).lower().rstrip('.')=='red')
    # Abort a live stream, then prove the engine still serves another request.
    try:
        payload=dict(model=a.model,messages=user('Count upward from one, one number per line, continuing as long as possible.'),max_tokens=4096,stream=True,temperature=0,chat_template_kwargs={'enable_thinking':False})
        req=urllib.request.Request(a.endpoint.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        seen=False
        with urllib.request.urlopen(req,timeout=60) as r:
            for line in r:
                if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
                    d=json.loads(line[6:]);delta=d.get('choices',[{}])[0].get('delta',{})
                    if delta.get('content'):seen=True;break
        report['results'].append(dict(name='cancel_stream',pass_=seen))
    except Exception as exc:report['results'].append(dict(name='cancel_stream',pass_=False,error=str(exc)))
    run('after_cancel',user('Reply with exactly RECOVERED.'),lambda m:text(m)=='RECOVERED')
    report['status']='pass' if all(x['pass_'] for x in report['results']) else 'fail'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='pass' else 1

if __name__=='__main__':raise SystemExit(main())
