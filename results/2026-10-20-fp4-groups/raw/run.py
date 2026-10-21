# SPDX-License-Identifier: AGPL-3.0-only
import concurrent.futures as cf
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent
CANDIDATE = json.loads((ROOT / 'candidate-image.json').read_text())
RANKS = json.loads((ROOT.parent / 'glm-rc-fp4-decode-20261020/model-plan.json').read_text())['ranks']
STAGE = '/home/enntitysparkadmin/glm-fp4-groups-20261020'
OWNER = 'codex-glm-fp4-groups'
owned, clocks = {}, set()

def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + '\n')

def ssh(host, args, data=None, check=True, timeout=300):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15',
        host, shlex.join(args)], input=data, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(host + ' ' + repr(args[:3]) + ': ' + p.stderr.decode()[-2000:] + p.stdout.decode()[-1000:])
    return p

def lloom(args):
    return ssh('ennspark01', ['env', 'LLOOM_OWNER='+OWNER, 'lloom'] + args)

def parallel(fn):
    with cf.ThreadPoolExecutor(2) as pool:
        return list(pool.map(fn, RANKS))

def stage(r):
    h = r['host']
    assert not ssh(h, ['docker', 'ps', '-q']).stdout.strip()
    assert ssh(h, ['systemctl', 'show', 'gpu-clock-cap.service', '--property=ActiveState', '--value']).stdout.strip() == b'active'
    ssh(h, ['mkdir', STAGE])
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as archive:
        for name in ['start-node.sh','probe.py','bench.py','prompts.json','rc4-baseline.json','shared-baseline.json','control.json','shared.json','dense.json','both.json','cases.json','grader.py','check_code.py']:
            archive.add(ROOT/name, arcname=name)
    ssh(h, ['tar', '-xf', '-', '-C', STAGE], data.getvalue())
    clocks.add(h)
    ssh(h, ['sudo', '-n', '/usr/bin/systemctl', 'stop', 'gpu-clock-cap.service'])
    assert ssh(h, ['systemctl', 'show', 'gpu-clock-cap.service', '--property=ActiveState', '--value']).stdout.strip() == b'inactive'

def start(r, arm):
    h = r['host']; name = 'atlas-sparkglm-fp4-groups-rank'+str(r['rank'])
    assert not ssh(h, ['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$']).stdout.strip()
    args = ['bash', STAGE+'/start-node.sh', '--mode', 'spark', '--rank', str(r['rank']),
        '--leader-address', r['leader'], '--model-root', r['model_root'],
        '--image', CANDIDATE['image_id'], '--name', name, '--profile', STAGE+'/'+arm+'.json',
        '--cuda-cache', '/home/enntitysparkadmin/glm-tp2-modes-20261020-validation/cuda-cache',
        '--fabric-interface', r['iface'], '--no-fabric-siblings', '--fabric-hca', r['hca'],
        '--gpu-memory-utilization', '0.91', '--world-size', '2', '--kv-shard', '--display-carveout',
        '--prefix-cache-dir', STAGE+'/prefix-cache-'+arm, '--prefix-cache-gb', '48']
    save(h+'-'+arm+'-launch.json', args)
    ssh(h, args)
    owned[h] = ssh(h, ['docker', 'inspect', '--format', '{{.Id}}', name]).stdout.decode().strip()
    print(h, 'started', flush=True)

def cleanup(r, restore_clocks=True):
    h = r['host']
    if h in owned:
        cid=owned[h]; info=json.loads(ssh(h, ['docker', 'inspect', cid]).stdout)[0]
        assert info['Image']==CANDIDATE['image_id'] and info['Name'].startswith('/atlas-sparkglm-fp4-groups-')
        p=ssh(h, ['docker', 'logs', cid], check=False)
        (ROOT/(h+'-'+current_arm+'-server.log')).write_bytes(p.stdout+p.stderr)
        ssh(h, ['docker', 'rm', '-f', cid]); del owned[h]
    if restore_clocks and h in clocks:
        ssh(h, ['sudo', '-n', '/usr/bin/systemctl', 'start', 'gpu-clock-cap.service'])
        assert ssh(h, ['systemctl', 'show', 'gpu-clock-cap.service', '--property=ActiveState', '--value']).stdout.strip()==b'active'
        clocks.remove(h)


def attest(r,arm):
    code="import pathlib,json,hashlib; rows=[]\nfor p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):\n try:\n  a=p.read_bytes().split(b'\\0')\n  if a and a[0]==b'/usr/local/bin/spark':\n   e=dict(s.decode().split('=',1) for s in (p.parent/'environ').read_bytes().split(b'\\0') if b'=' in s); rows.append({'argv':[s.decode() for s in a if s], 'atlas_env':{k:v for k,v in e.items() if k.startswith('ATLAS_')}})\n except OSError:pass\nprint(json.dumps({'processes':rows,'source':json.loads(pathlib.Path('/opt/atlas/source-manifest.json').read_text()),'binary_sha256':hashlib.sha256(pathlib.Path('/usr/local/bin/spark').read_bytes()).hexdigest()}))"
    x=json.loads(ssh(r['host'],['docker','exec',owned[r['host']],'python3','-c',code]).stdout)
    assert x['binary_sha256']==CANDIDATE['binary_sha256'] and len(x['processes'])==1
    e=x['processes'][0]['atlas_env'];p=json.loads((ROOT/(arm+'.json')).read_text())['environment']
    for flag in ('ATLAS_GLM_FP4L_SHARED_FP8','ATLAS_GLM_FP4L_DENSE_MXFP8'):assert e[flag]==p[flag]
    assert e['ATLAS_GLM_SERVING_MODE']=='spark' and e['ATLAS_GLM_FP4L_KDA_MXFP8_DECODE']=='0'
    assert e['ATLAS_GLM_MLA_FUSE']=='1' and e['ATLAS_GLM_FP4L_MLA_FP8']=='1'
    x['clocks']=ssh(r['host'],['nvidia-smi','--query-gpu=clocks.gr,clocks.max.gr,power.draw','--format=csv,noheader']).stdout.decode()
    save(r['host']+'-'+arm+'-identity.json',x)

results={}; decisions={};current_arm='control';succeeded=False
job=json.loads((ROOT/'job.json').read_text())['id']
def eligible(arm,baseline):
    import math
    x,b=results[arm],results[baseline]
    assert x['requests']==b['requests']
    assert [r['request_sha256'] for r in x['quality']]==[r['request_sha256'] for r in b['quality']]
    ratios={n:x['decode'][n]['decode_tokens_per_second']['median']/b['decode'][n]['decode_tokens_per_second']['median'] for n in x['decode']}
    overall=math.prod(ratios.values())**(1/len(ratios))
    accepted=x['quality_passed'] and x['speed_finished'] and overall>=1.02 and min(ratios.values())>=0.98
    d={'baseline':baseline,'ratios':ratios,'geomean_ratio':overall,'quality_passed':x['quality_passed'],'accepted':accepted,'rule':'six quality passes; all timing gates; geometric mean gain >=2%; no workload regression >2%'}
    decisions[arm]=d;save('decisions.json',decisions);return accepted
try:
    parallel(stage)
    for index in range(3):
        current_arm=['control','shared','both' if decisions.get('shared',{}).get('accepted') else 'dense'][index]
        for r in reversed(RANKS):start(r,current_arm)
        deadline=time.monotonic()+600
        while ssh('ennspark01',['curl','-fsS','--max-time','2','http://127.0.0.1:8893/health'],check=False).returncode:
            assert time.monotonic()<deadline,'startup timeout'
            for r in RANKS:assert ssh(r['host'],['docker','inspect','--format','{{.State.Running}}',owned[r['host']]]).stdout.strip()==b'true','container exited'
            time.sleep(3)
        parallel(lambda r:attest(r,current_arm))
        lloom(['job','note',job,current_arm+': ready; six timed requests and six quality cases'])
        print(current_arm,'ready',flush=True)
        with (ROOT/(current_arm+'-probe.log')).open('wb') as log:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=15','ennspark01',shlex.join(['python3',STAGE+'/probe.py',current_arm])],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        raw=ssh('ennspark01',['cat',STAGE+'/'+current_arm+'-result.json'],check=False)
        if raw.returncode==0:
            results[current_arm]=json.loads(raw.stdout);save(current_arm+'-result.json',results[current_arm])
        assert p.returncode==0,'probe failed; see log'
        if index:eligible(current_arm,'shared' if current_arm=='both' else 'control')
        print(current_arm,'done',json.dumps(decisions.get(current_arm,{'quality_passed':results[current_arm]['quality_passed']})),flush=True)
        parallel(lambda r:cleanup(r,False))
    selected='both' if decisions.get('both',{}).get('accepted') else ('shared' if decisions['shared']['accepted'] else ('dense' if decisions.get('dense',{}).get('accepted') else 'control'))
    save('selected.json',{'arm':selected,'flags':{k:v for k,v in json.loads((ROOT/(selected+'.json')).read_text())['environment'].items() if k in ('ATLAS_GLM_FP4L_SHARED_FP8','ATLAS_GLM_FP4L_DENSE_MXFP8')}})
    succeeded=True;print('Selected:',selected,flush=True)
except BaseException as e:
    save('error.json',{'error':repr(e)});raise
finally:
    errors=[]
    for r in RANKS:
        try:cleanup(r)
        except Exception as e:errors.append(repr(e))
    save('cleanup.json',{'errors':errors,'owned':owned,'clocks':list(clocks)})
    if not succeeded and not errors:
        restored=lloom(['release','pair','--apply','--yes','--json']);save('release.json',json.loads(restored.stdout));print('Restored prior RC',flush=True)
