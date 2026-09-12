import pathlib,time,json,subprocess
p=pathlib.Path('/tmp/sg-cache12');low=0;seen=False
with (p/'memory.jsonl').open('a',buffering=1) as f:
 for i in range(3600):
  state=subprocess.run(['docker','inspect','-f','{{.State.Running}}','sg-cache12'],capture_output=True,text=True).stdout.strip()
  if state=='true':seen=True
  elif seen:break
  m={a:int(b.split()[0])*1024 for a,b in (x.split(':',1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines())}
  f.write(json.dumps({'time':time.time(),'available':m['MemAvailable'],'swap_used':m['SwapTotal']-m['SwapFree'],'running':state=='true'})+'\n')
  low=low+1 if state=='true' and m['MemAvailable']<1.5*1024**3 else 0
  if low>=3:
   (p/'MEMORY_ABORT').write_text('MemAvailable below 1.5 GiB for three 2-second samples; stopped experimental container.\n');subprocess.run(['docker','stop','-t','3','sg-cache12']);break
  time.sleep(2)
