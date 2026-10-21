# SPDX-License-Identifier: AGPL-3.0-only
import json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
deadline=time.monotonic()+600
while not all((r/f).exists() for f in ['transfer.json','gpu-tests.json']):
    assert time.monotonic()<deadline,'build/test readiness timeout'
    if (r/'gpu-tests.json').exists():assert json.loads((r/'gpu-tests.json').read_text())['returncode']==0
    time.sleep(2)
assert json.loads((r/'gpu-tests.json').read_text())['returncode']==0
subprocess.run(['python3',str(r/'run.py')],check=True)
