#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Narrow GLM runtime update; credentials and rollback configs remain on each host."""
import argparse, json, pathlib, shlex, subprocess

ROOT = pathlib.Path(__file__).resolve().parent
CANDIDATE = json.loads((ROOT/'candidate-image.json').read_text())
IMAGE = CANDIDATE['image_id']
REMOTE = r'''
import json,pathlib,subprocess,sys,hashlib,os,datetime
mode,host,image,binary,commit=sys.argv[1:]
home=pathlib.Path.home();live=home/'.lloom/config.json'
root=home/'.lloom/releases/glm-mla-fusion-20261020'
raw=live.read_bytes();cfg=json.loads(raw)
assert cfg['cluster']['nodeId']==host
ids=['glm53-flash-atlas-head','glm53-flash-atlas-worker']
def digest(b):return hashlib.sha256(b).hexdigest()
def run(*args):return subprocess.check_output(args,text=True).strip()
env=dict(os.environ);env['PATH']=':'.join([str(home/'.local/bin'),*[str(p) for p in sorted((home/'.local/opt').glob('node-*/bin'),reverse=True)],env.get('PATH','')])
if mode=='inspect':
 im=json.loads(run('docker','image','inspect',image))[0]
 containers=json.loads(run('docker','inspect','lloom-atlas-sparkglm-'+host))
 print(json.dumps({'host':host,'config_sha256':digest(raw),'candidate':{k:im.get(k) for k in ['Id','RepoTags','RepoDigests','Created']},'candidate_entrypoint':im['Config']['Entrypoint'],'current_container':[{k:c[k] for k in ['Id','Image','State']} for c in containers],'runtimes':{i:{'image':cfg['runtimes'][i]['bootstrap']['image'],'node':cfg['runtimes'][i].get('node'),'profile':[x for x in cfg['runtimes'][i]['bootstrap']['createArgs'] if x.startswith('SPARKGLM_PROFILE=')]} for i in ids},'maintenance':cfg['runtimes']['glm53-flash-atlas-cluster'].get('maintenance')}))
elif mode=='stage':
 assert root.exists() is False
 group=cfg['runtimes']['glm53-flash-atlas-cluster']
 if host=='ennspark01':assert group.get('maintenance',{}).get('state')=='suspended'
 root.mkdir(mode=0o700)
 (root/'before.json').write_bytes(raw);(root/'before.json').chmod(0o600)
 staged=json.loads(raw);changes=[]
 for i in ids:
  b=staged['runtimes'][i]['bootstrap'];old=b['image']
  assert old=='sha256:3de7af108f595066b5e05b1dd5c171ebbc534e9dd5c6cb80f2e58087e5e44d3c', old
  assert b['pull'] is False and 'SPARKGLM_MODE=spark' in b['createArgs']
  b['image']=image
  changes.append({'runtime':i,'image_before':old,'image_after':image})
 data=(json.dumps(staged,indent=2)+'\n').encode()
 (root/'stage.json').write_bytes(data);(root/'stage.json').chmod(0o600)
 print(json.dumps({'host':host,'before_sha256':digest(raw),'stage_sha256':digest(data),'backup':str(root/'before.json'),'changes':changes,'maintenance_preserved':True}))
elif mode=='publish':
 # Use LLooM's atomic writer with strict external-change checks; serialize these publishes.
 code=r"""
import fs from 'node:fs/promises';import fss from 'node:fs';import assert from 'node:assert/strict';import crypto from 'node:crypto';import {pathToFileURL} from 'node:url';
const [home,root,expected]=process.argv.slice(1),live=home+'/.lloom/config.json';
const before=await fs.readFile(root+'/before.json'),stage=JSON.parse(await fs.readFile(root+'/stage.json','utf8'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const old=JSON.parse(before);assert.equal(sha(await fs.readFile(live)),sha(before),'Config changed since staging');
const restored=structuredClone(stage);
for(const id of ['glm53-flash-atlas-head','glm53-flash-atlas-worker']){
 assert.equal(stage.runtimes[id].bootstrap.image,expected);
 restored.runtimes[id].bootstrap.image=old.runtimes[id].bootstrap.image;
}
assert.deepEqual(restored,old,'Unexpected changes outside GLM image pins');
const {mutateConfigSource}=await import(pathToFileURL(home+'/.local/lib/node_modules/lloom/src/config-mutation.mjs'));
await mutateConfigSource({sourcePath:live},current=>{
 assert.equal(sha(fss.readFileSync(live)),sha(before),'Concurrent config edit');
 assert.deepEqual(current,old);for(const k of Object.keys(current))delete current[k];Object.assign(current,stage);
});
assert.deepEqual(JSON.parse(await fs.readFile(live,'utf8')),stage);
console.log(JSON.stringify({published:true,config_sha256:sha(await fs.readFile(live)),image:expected}));
"""
 result=subprocess.run(['node','--input-type=module','-e',code,str(home),str(root),image],env=env,capture_output=True,text=True)
 if result.returncode:raise RuntimeError('Guarded publish failed: '+result.stderr[-1200:])
 print(result.stdout)
elif mode=='identity':
 name='lloom-atlas-sparkglm-'+host
 inspect=json.loads(run('docker','inspect',name))[0]
 assert inspect['Image']==image
 code=r"""import pathlib,json,hashlib
a=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  v=p.read_bytes().split(b'\0');e=dict(s.decode().split('=',1) for s in (p.parent/'environ').read_bytes().split(b'\0') if b'=' in s)
  if v and v[0]==b'/usr/local/bin/spark':a.append({'argv':[s.decode() for s in v if s],'env':{k:v for k,v in e.items() if k.startswith(('ATLAS_','SPARKGLM_','NCCL_SOCKET','NCCL_IB_HCA'))}})
 except (OSError,PermissionError):pass
print(json.dumps({'processes':a,'manifest':json.loads(pathlib.Path('/opt/atlas/source-manifest.json').read_text()),'binary_sha256':hashlib.sha256(pathlib.Path('/usr/local/bin/spark').read_bytes()).hexdigest(),'profile':json.loads(pathlib.Path('/opt/atlas/profiles/4x1m.json').read_text())}))"""
 data=json.loads(run('docker','exec',name,'python3','-c',code))
 assert data['binary_sha256']==binary
 assert len(data['processes'])==1
 e=data['processes'][0]['env']
 assert e['ATLAS_GLM_MLA_FUSE']=='1'
 assert e['ATLAS_GLM_FP4L_NATIVE']=='1'
 assert e['ATLAS_GLM_FP4L_KDA_MXFP8_DECODE']=='0'
 assert e.get('ATLAS_GLM_FP4L_KDA_TILED_TWIN','0')=='0'
 assert e.get('ATLAS_DFLASH_TP2_LINEAR_TWO_ROWS','0')=='0'
 assert data['manifest']['commit']==commit
 assert e['ATLAS_GLM_SERVING_MODE']=='spark' and e['SPARKGLM_MODE']=='spark'
 assert e['ATLAS_GLM_FP4L_DECODE_A16']=='0'
 print(json.dumps({'host':host,'image':inspect['Image'],'container':name,'state':inspect['State'],**data}))
'''
p = argparse.ArgumentParser()
p.add_argument("mode", choices=["inspect", "stage", "publish", "identity"])
p.add_argument("host", choices=["ennspark01", "ennspark02"])
a = p.parse_args()
out = ROOT / (a.mode + "-" + a.host + ".json")
assert not out.exists(), out
r = subprocess.run(["ssh","-o","BatchMode=yes",a.host,shlex.join(["python3","-c",REMOTE,a.mode,a.host,IMAGE,CANDIDATE['binary_sha256'],CANDIDATE['source']['commit']])],capture_output=True,text=True)
if r.returncode:
    raise RuntimeError(r.stderr[-2500:])
data = json.loads(r.stdout)
out.write_text(json.dumps(data,indent=2)+"\n")
out.chmod(0o600)
print(json.dumps(data if a.mode != "identity" else {k:data[k] for k in ["host","image","container","binary_sha256"]},indent=2))
