#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Top-8 cardinality follow-up; initial universal-dispatch screen stays failed."""
import argparse, hashlib, json
from pathlib import Path
from bench import case
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--iterations', type=int, default=40)
p.add_argument('--repeats', type=int, default=7)
a = p.parse_args()
output = Path(a.output)
if output.exists(): raise SystemExit('output exists')
r = dict(schema='sparkglm.e3-policy-screen/v1', status='running', results=[],
         source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in Path(__file__).parent.glob('*.py')})
try:
    for tokens in (2048, 4096, 7168):
        # Balanced real top-8 cardinality, plus skew with the same route count.
        n, remainder = divmod(tokens * 8, 288)
        balanced = [n + int(i < remainder) for i in range(288)]
        skewed = [0] * 144 + [2*n + int(i < 2*remainder) for i in range(144)]
        # Distribute any remainder exactly without changing cardinality.
        skewed[-1] += tokens * 8 - sum(skewed)
        for name, counts in [('balanced', balanced), ('skewed', skewed)]:
            result = case(counts, 20260908, a.iterations, a.repeats, 128, tokens)
            result['routing'] = name
            r['results'].append(result)
            print(json.dumps(dict(tokens=tokens,routing=name,speedup=result['speedup'])),flush=True)
            output.write_text(json.dumps(r,indent=2)+'\n')
    r['status']='passed-correctness'
except Exception as exc:
    r['status']='failed';r['error']=str(exc);raise
finally: output.write_text(json.dumps(r,indent=2)+'\n')
