# SPDX-License-Identifier: AGPL-3.0-only
"""Package only the accepted Spark precision defaults, keeping the tested binary."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
PRODUCT=ROOT.parent/'glm-native-mtp-20261020/product'
selection=json.loads((ROOT/'selected.json').read_text())
assert selection['arm']!='control','No candidate qualified'
for arm in ('control','shared',selection['arm']):
    result=json.loads((ROOT/(arm+'-result.json')).read_text())
    assert result['speed_finished'], f'{arm} has an incomplete timing baseline'
assert json.loads((ROOT/(selection['arm']+'-result.json')).read_text())['quality_passed']
flags=selection['flags']
candidate=json.loads((ROOT/'candidate-image.json').read_text())

def git(*args):
    return subprocess.check_output(['git','-C',str(PRODUCT),*args],text=True).strip()

path=PRODUCT/'install/serve.py'
s=path.read_text()
for name,value in flags.items():
    old=f"        '{name}': '1',"
    new=f"        '{name}': '1' if mode == 'strict' else '{value}',"
    assert old in s;s=s.replace(old,new,1)
s=s.replace("value = profile['environment'].get(flag, '1')","value = profile['environment'].get(flag, required[flag])")
# The public preset only accepts conversions that qualified. Native opt-outs
# remain available for accepted groups; rejected groups remain forced native.
allowed=tuple(k for k,v in flags.items() if v=='0')
s=s.replace("for flag in ('ATLAS_GLM_FP4L_SHARED_FP8', 'ATLAS_GLM_FP4L_DENSE_MXFP8'):", f"for flag in {allowed!r}:", 1)
path.write_text(s)
path=PRODUCT/'install/test_modes.py';s=path.read_text()
s=s.replace("for flag in ('ATLAS_GLM_FP4L_SHARED_FP8', 'ATLAS_GLM_FP4L_DENSE_MXFP8'):", f"for flag in {allowed!r}:", 1)
rejected=tuple(k for k,v in flags.items() if v=='1')
if rejected:
    marker='    def test_modes_are_explicit_and_do_not_mutate_the_capacity_profile(self):'
    test='    def test_unqualified_group_stays_native_in_spark(self):\n'+f'        for flag in {rejected!r}:\n'+"            self.profile['environment'][flag] = '0'\n            with self.assertRaises(ValueError):\n                self.launch('spark')\n            del self.profile['environment'][flag]\n\n"
    assert marker in s;s=s.replace(marker,test+marker)
s=s.replace('test_spark_allows_only_the_measured_kda_target_conversion','test_spark_uses_qualified_target_conversions')
s=s.replace('("MLA_FP8", "SHARED_FP8", "DENSE_MXFP8", "KDA_W8A8_PREFILL")','("MLA_FP8", "KDA_W8A8_PREFILL")')
needle='        self.assertEqual(env["ATLAS_GLM_FP4L_DECODE_A16"], "0")'
addition=''.join(f'        self.assertEqual(env["{k}"], "{v}")\n' for k,v in flags.items())
assert needle in s;s=s.replace(needle,addition+needle,1)
needle='        self.assertEqual(env["ATLAS_GLM_FP4L_INPUT_SCALES"], "1")'
s=s.replace(needle,needle+'\n'+''.join(f'        self.assertEqual(env["{k}"], "1")\n' for k in flags).rstrip(),1)
path.write_text(s)
path=PRODUCT/'README.md';s=path.read_text()
old='''Spark retains the optimized serving profile, calibrated A4 routed decode and
DFlash, and adds round-to-nearest-even FP4 KDA decode weights. Native KDA prefill,
MLA, shared experts, dense projections and the vocabulary head stay in their'''
converted=['KDA decode']+[label for k,label in [('ATLAS_GLM_FP4L_SHARED_FP8','shared-expert'),('ATLAS_GLM_FP4L_DENSE_MXFP8','dense-MLP')] if flags[k]=='0']
native=['KDA prefill','MLA','the vocabulary head']+[label for k,label in [('ATLAS_GLM_FP4L_SHARED_FP8','shared experts'),('ATLAS_GLM_FP4L_DENSE_MXFP8','dense projections')] if flags[k]=='1']
new='Spark retains the optimized serving profile, calibrated A4 routed decode and\nDFlash, and uses round-to-nearest-even FP4 '+', '.join(converted)+' weights.\n'+', '.join(native)+' stay in their'
assert old in s;s=s.replace(old,new)
path.write_text(s)
with (ROOT/'product-tests.log').open('w') as log:
    subprocess.run(['python3','-m','unittest','discover','-s',str(PRODUCT/'install'),'-p','test_*.py'],stdout=log,stderr=subprocess.STDOUT,check=True)
git('diff','--check')
git('add','install/serve.py','install/test_modes.py','install/atlas-source.json','README.md')
git('commit','-m','sparkglm: enable qualified corrected FP4 groups in Spark mode')
tree=git('rev-parse','HEAD:install');tag='localhost/atlas-sparkglm:fp4-groups-'+tree[:12]
context=ROOT/'final-context';context.mkdir()
shutil.copy2(PRODUCT/'install/serve.py',context/'serve.py')
(context/'Dockerfile').write_text(f'''FROM {candidate['tag']}
COPY serve.py /opt/atlas/serve.py
LABEL io.enntity.sparkglm.install-tree="{tree}"
''')
data=io.BytesIO()
with tarfile.open(fileobj=data,mode='w') as tf:
    for p in context.iterdir():tf.add(p,arcname=p.name)
stage='/home/enntitysparkadmin/glm-fp4-groups-20261020-final'
def ssh(host,args,**kwargs):
    return subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(args)],check=True,**kwargs)
ssh('ennspark01',['mkdir',stage]);ssh('ennspark01',['tar','-xf','-','-C',stage],input=data.getvalue())
with (ROOT/'final-build.log').open('wb') as log:
    ssh('ennspark01',['docker','build','--network=none','--pull=false','-t',tag,stage],stdout=log,stderr=subprocess.STDOUT)
image=ssh('ennspark01',['docker','image','inspect','--format','{{.Id}}',tag],capture_output=True,text=True).stdout.strip()
ssh('ennspark01',['bash','-c','set -o pipefail; docker save '+shlex.quote(tag)+' | ssh -o BatchMode=yes ennspark02-fab docker load'],stdout=subprocess.DEVNULL)
expected=hashlib.sha256((context/'serve.py').read_bytes()).hexdigest()
code="import hashlib,pathlib,json;print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ['/usr/local/bin/spark','/opt/atlas/serve.py']}))"
def verify(host):
    identity=ssh(host,['docker','image','inspect','--format','{{.Id}}',tag],capture_output=True,text=True).stdout.strip();assert identity==image
    got=json.loads(ssh(host,['docker','run','--rm','--runtime=runc','--network=none','--memory','128m','--entrypoint','python3',image,'-c',code],capture_output=True,text=True).stdout)
    assert got['/usr/local/bin/spark']==candidate['binary_sha256'] and got['/opt/atlas/serve.py']==expected
    return {'host':host,'image':identity,'hashes':got}
with concurrent.futures.ThreadPoolExecutor(2) as pool:verified=list(pool.map(verify,['ennspark01','ennspark02']))
candidate.update(image_id=image,tag=tag,install_tree=tree,product_commit=git('rev-parse','HEAD'),selected=selection,verified=verified)
(ROOT/'final-image.json').write_text(json.dumps(candidate,indent=2)+'\n')
print(json.dumps({'image':image,'selected':selection,'binary_unchanged':True}))
