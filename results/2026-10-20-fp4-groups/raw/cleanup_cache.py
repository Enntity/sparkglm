# SPDX-License-Identifier: AGPL-3.0-only
import concurrent.futures,json,shlex,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
image=json.loads((r/'candidate-image.json').read_text())['image_id']
code="import pathlib,shutil,json; p=pathlib.Path('/scratch'); names=[x.name for x in p.iterdir()]; [(shutil.rmtree(x) if x.is_dir() and not x.is_symlink() else x.unlink()) for x in p.iterdir()]; print(json.dumps({'removed':names,'empty':not any(p.iterdir())}))"
def one(host):
 rows=[]
 for arm in [p.name.removesuffix('-result.json') for p in r.glob('*-result.json') if p.name in ('control-result.json','shared-result.json','dense-result.json','both-result.json')]:
  path='/home/enntitysparkadmin/glm-fp4-groups-20261020/prefix-cache-'+arm
  args=['docker','run','--rm','--runtime=runc','--network','none','--memory','128m','--memory-swap','128m','--entrypoint','python3','--mount','type=bind,src='+path+',dst=/scratch',image,'-c',code]
  result=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes',host,shlex.join(args)],text=True));assert result['empty']
  subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(['rmdir',path])],check=True)
  rows.append({'path':path,**result})
 return {'host':host,'cleanup':rows}
with concurrent.futures.ThreadPoolExecutor(2) as p:rows=list(p.map(one,['ennspark01','ennspark02']))
(r/'cache-cleanup.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Only task-owned temporary prefix caches removed on both nodes.')
