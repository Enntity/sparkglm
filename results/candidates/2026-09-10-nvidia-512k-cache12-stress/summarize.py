# SPDX-License-Identifier: Apache-2.0
# Original SparkGLM measurement summarizer.
import pathlib,json,statistics,sys
r=pathlib.Path(sys.argv[1]);out={}
events=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines()] if (r/'events.jsonl').exists() else []
out['events']=events
for role in ('head','worker'):
 p=r/f'{role}-memory.jsonl'
 if not p.exists():continue
 allrows=[json.loads(l) for l in p.read_text().splitlines()];rows=[x for x in allrows if x['running']]
 d={'minimum_available_gib':min(x['available'] for x in rows)/2**30,'maximum_swap_occupied_gib':max(x['swap_used'] for x in rows)/2**30,'maximum_full_pressure_avg10':max(x['pressure']['full']['avg10'] for x in rows)}
 ready=next((x['time'] for x in events if x['stage']=='READY'),None)
 end=json.loads((r/'cancellation.json').read_text())['completed_at']
 for stage,part in [('all',rows),('serving',[x for x in rows if ready and ready<=x['time']<=end])]:
  if len(part)<2:continue
  first,last=part[0],part[-1];d[stage]={'duration_s':last['time']-first['time'],'swap_in_gib':(last['pswpin_pages']-first['pswpin_pages'])*4096/2**30,'swap_out_gib':(last['pswpout_pages']-first['pswpout_pages'])*4096/2**30,'memory_full_stall_seconds':(last['pressure']['full']['total']-first['pressure']['full']['total'])/1e6,'minimum_available_gib':min(x['available'] for x in part)/2**30}
 out[role]=d
runs=[json.loads(p.read_text()) for p in sorted(r.glob('c4-r[123].json'))]
if runs:out['c4']={'wall_s':[x['wall_s'] for x in runs],'median_wall_s':statistics.median(x['wall_s'] for x in runs),'requests_ok':all(z['error'] is None and z['completion_tokens']==400 for x in runs for z in x['requests'])}
for name in ['near512','longc4-0','longc4-1','longc4-2','longc4-3']:
 p=r/(name+'.json')
 if p.exists():
  x=json.loads(p.read_text());out[name]={'tokens':x['tokenized_prompt_tokens'],'ttft_s':[t['ttft_s'] for t in x['turns']],'correct':[t['correct'] for t in x['turns']]}
p=r/'semantics.json'
if p.exists():
 x=json.loads(p.read_text());out['semantics']={'passed':sum(t['pass_'] for t in x['results']),'failed':[t['name'] for t in x['results'] if not t['pass_']]}
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
