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
STAGE = '/home/enntitysparkadmin/glm-mla-fusion-20261020'
OWNER = 'codex-glm-mla-fusion'
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
        for name in ['start-node.sh', 'probe.py', 'bench.py', 'prompts.json', 'rc4-baseline.json', 'shared-baseline.json','parity.py','off.json','on.json']:
            archive.add(ROOT/name, arcname=name)
    ssh(h, ['tar', '-xf', '-', '-C', STAGE], data.getvalue())
    clocks.add(h)
    ssh(h, ['sudo', '-n', '/usr/bin/systemctl', 'stop', 'gpu-clock-cap.service'])
    assert ssh(h, ['systemctl', 'show', 'gpu-clock-cap.service', '--property=ActiveState', '--value']).stdout.strip() == b'inactive'

def start(r, arm):
    h = r['host']; name = 'atlas-sparkglm-mla-fuse-rank'+str(r['rank'])
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
        assert info['Image']==CANDIDATE['image_id'] and info['Name'].startswith('/atlas-sparkglm-mla-fuse-')
        p=ssh(h, ['docker', 'logs', cid], check=False)
        (ROOT/(h+'-'+current_arm+'-server.log')).write_bytes(p.stdout+p.stderr)
        ssh(h, ['docker', 'rm', '-f', cid]); del owned[h]
    if restore_clocks and h in clocks:
        ssh(h, ['sudo', '-n', '/usr/bin/systemctl', 'start', 'gpu-clock-cap.service'])
        assert ssh(h, ['systemctl', 'show', 'gpu-clock-cap.service', '--property=ActiveState', '--value']).stdout.strip()==b'active'
        clocks.remove(h)

current_arm='off'
succeeded=False
try:
    parallel(stage)
    for current_arm in ['off', 'on']:
        for r in reversed(RANKS): start(r, current_arm)
        deadline=time.monotonic()+600
        while ssh('ennspark01', ['curl','-fsS','--max-time','2','http://127.0.0.1:8893/health'],check=False).returncode:
            assert time.monotonic()<deadline,'startup timeout'
            time.sleep(3)
        for r in RANKS:
            h=r['host']; cid=owned[h]
            code="import pathlib,json,hashlib; rows=[]\nfor p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):\n try:\n  a=p.read_bytes().split(b'\\0')\n  if a and a[0]==b'/usr/local/bin/spark':\n   e=dict(s.decode().split('=',1) for s in (p.parent/'environ').read_bytes().split(b'\\0') if b'=' in s); rows.append({'argv':[s.decode() for s in a if s], 'atlas_env':{k:v for k,v in e.items() if k.startswith('ATLAS_')}})\n except OSError:pass\nprint(json.dumps({'processes':rows,'source':json.loads(pathlib.Path('/opt/atlas/source-manifest.json').read_text()),'binary_sha256':hashlib.sha256(pathlib.Path('/usr/local/bin/spark').read_bytes()).hexdigest()}))"
            ident=json.loads(ssh(h,['docker','exec',cid,'python3','-c',code]).stdout)
            assert ident['binary_sha256']==CANDIDATE['binary_sha256']
            assert ident['processes'][0]['atlas_env']['ATLAS_GLM_SERVING_MODE']=='spark'
            assert ident['processes'][0]['atlas_env']['ATLAS_GLM_MLA_FUSE'] == ('0' if current_arm=='off' else '1')
            save(h+'-'+current_arm+'-identity.json',ident)
        with (ROOT/('parity-'+current_arm+'.log')).open('wb') as log:
            p=subprocess.run(['ssh','-o','BatchMode=yes','ennspark01',shlex.join(['python3',STAGE+'/parity.py',current_arm])],stdout=log,stderr=subprocess.STDOUT,timeout=240)
        raw=ssh('ennspark01',['cat',STAGE+'/parity-'+current_arm+'.json'])
        save('parity-'+current_arm+'.json',json.loads(raw.stdout))
        assert p.returncode==0,'parity probe failed'
        if current_arm=='off':
            parallel(lambda r:cleanup(r,False))
    off=json.loads((ROOT/'parity-off.json').read_text());on=json.loads((ROOT/'parity-on.json').read_text())
    exact={name:all(off[name][k]==on[name][k] for k in ['token_ids','content','reasoning_content','finish_reason']) for name in off}
    save('parity-summary.json',{'exact':exact,'tokens':sum(len(x['token_ids']) for x in on.values())})
    assert all(exact.values()),'fusion changed full-model outputs'
    lloom(['job','note','j-5762f37a','GPU byte parity and five full-model output comparisons passed; timing the same nine uncapped requests'])
    print('Spark RC ready: nine decode requests only',flush=True)
    with (ROOT/'probe.log').open('wb') as log:
        p=subprocess.run(['ssh','-o','BatchMode=yes','ennspark01',shlex.join(['python3',STAGE+'/probe.py'])],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    raw=ssh('ennspark01',['cat',STAGE+'/result.json'],check=False)
    if raw.returncode==0:save('result.json',json.loads(raw.stdout))
    assert p.returncode==0,'probe failed; see probe.log'
    succeeded=True
    print('Nine requests complete; cleaned pair will stay reserved for the RC decision',flush=True)
except BaseException as error:
    save('error.json',{'error':repr(error)})
    raise
finally:
    errors=[]
    for r in RANKS:
        try:cleanup(r)
        except Exception as error:errors.append(repr(error))
    save('cleanup.json',{'errors':errors,'owned':owned,'clocks':list(clocks)})
    if not succeeded and not errors and (ROOT/'reservation.json').exists():
        restored=lloom(['release','pair','--apply','--yes','--json'])
        save('release.json',json.loads(restored.stdout))
        print('Pair released; normal serving restored',flush=True)
