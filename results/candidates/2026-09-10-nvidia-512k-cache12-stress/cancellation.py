# SPDX-License-Identifier: Apache-2.0
# Original SparkGLM cancellation measurement helper.
import json,time,pathlib,urllib.request,re
R=pathlib.Path('.');base='http://127.0.0.1:8895';out={'started_at':time.time()}
def metrics():return urllib.request.urlopen(base+'/metrics',timeout=10).read().decode()
def active(s):return sum(float(v) for v in re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})? ([0-9.e+]+)$',s,re.M))
def post(body):return urllib.request.urlopen(urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=90)
try:
 before=metrics();assert active(before)==0; (R/'cancel-before.prom').write_text(before)
 start=time.monotonic()
 with post({'model':'sparkglm-package','messages':[{'role':'user','content':'Write a very long numbered list of 2000 distinct everyday objects. Continue until all 2000 are listed.'}],'max_tokens':8192,'temperature':0,'stream':True,'chat_template_kwargs':{'enable_thinking':False}}) as response:
  for line in response:
   if line.startswith(b'data: {'):
    chunk=json.loads(line[6:]);delta=chunk.get('choices',[{}])[0].get('delta',{})
    if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning'):
     out['first_token_s']=time.monotonic()-start;out['closed_stream_at']=time.time();break
  else:raise RuntimeError('No token before stream ended')
 for i in range(40):
  time.sleep(.5);s=metrics()
  if active(s)==0:break
 else:raise RuntimeError('Cancelled request did not drain within 20 seconds')
 out['drain_after_close_s']=time.time()-out['closed_stream_at'];(R/'cancel-after.prom').write_text(s)
 for n in range(3):
  marker=f'APPLIANCE_RECOVERY_{n}_OK';t=time.monotonic()
  with post({'model':'sparkglm-package','messages':[{'role':'user','content':f'Reply with exactly {marker} and nothing else.'}],'max_tokens':256,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}) as response:x=json.load(response)
  content=x['choices'][0]['message']['content'];out.setdefault('recovery',[]).append({'wall_s':time.monotonic()-t,'content':content,'correct':content.strip()==marker});assert content.strip()==marker
 out['passed']=True
except Exception as e:out['passed']=False;out['error']=str(e)
finally:
 out['completed_at']=time.time();(R/'cancellation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if not out['passed']:raise SystemExit(1)
