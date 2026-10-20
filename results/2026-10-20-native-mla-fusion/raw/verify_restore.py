# SPDX-License-Identifier: AGPL-3.0-only
"""Verify the restored managed GLM image and one gateway response."""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
EXPECTED = json.loads((ROOT / "candidate-image.json").read_text())["image_id"]


def ssh(host, args):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host, shlex.join(args)], text=True
    )


def identity(host):
    rows = json.loads(ssh(host, ["docker", "inspect", "lloom-atlas-sparkglm-" + host]))
    row = rows[0]
    assert row["Image"] == EXPECTED and row["State"]["Running"]
    assert "SPARKGLM_MODE=spark" in row["Config"]["Env"]
    assert ssh(host, ["systemctl", "show", "gpu-clock-cap.service",
                     "--property=ActiveState", "--value"]).strip() == "active"
    return {"host": host, "image": row["Image"], "running": True,
            "mode": "spark", "clock_service": "active"}


with concurrent.futures.ThreadPoolExecutor(2) as pool:
    identities = list(pool.map(identity, ["ennspark01", "ennspark02"]))

code = r"""
import json,pathlib,urllib.request,time,os,re
cfg=json.loads((pathlib.Path.home()/'.lloom/config.json').read_text())
env=dict(os.environ)
envfile=pathlib.Path.home()/'.config/lloom/env'
if envfile.exists():
 for line in envfile.read_text().splitlines():
  match=re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$',line.strip())
  if match:
   key,value=match.groups();value=value.strip()
   if len(value)>=2 and value[0]==value[-1] and value[0] in ('"',"'"):value=value[1:-1]
   env.setdefault(key,value)
keys=cfg['security']['apiKeys']
headers={'Content-Type':'application/json'}
if keys:
 key=re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}',lambda m:env[m[1]],keys[0])
 headers['Authorization']='Bearer '+key
base='http://127.0.0.1:'+str(cfg['server']['port'])
def call(path,body=None):
 req=urllib.request.Request(base+path,json.dumps(body).encode() if body else None,headers)
 with urllib.request.urlopen(req,timeout=180) as r: return json.load(r)
model='glm-5.3-flash-atlas'
models=call('/v1/models')
assert any(x['id']==model for x in models['data'])
body={'model':model,'messages':[{'role':'user','content':'What is 17 + 25? Reply with only the integer.'}],
      'temperature':0,'max_tokens':128,'chat_template_kwargs':{'reasoning_effort':'low'}}
t=time.monotonic();answer=call('/v1/chat/completions',body)
assert answer['model']==model
c=answer['choices'][0]
assert c['finish_reason']=='stop' and c['message']['content'].strip()=='42'
print(json.dumps({'passed':True,'seconds':time.monotonic()-t,'request':body,'response':answer}))
"""
canary = json.loads(ssh("ennspark01", ["python3", "-c", code]))
(ROOT / "restored.json").write_text(
    json.dumps({"identities": identities, "gateway_canary": canary}, indent=2) + "\n"
)
print("New fusion RC and normal clocks verified on both ranks; gateway answered 42.")
