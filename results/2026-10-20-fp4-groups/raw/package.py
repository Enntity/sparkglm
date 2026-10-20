# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/glm-fp4-groups-20261020'
raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
selected=json.loads((ROOT/'selected.json').read_text())
decisions=json.loads((ROOT/'decisions.json').read_text())
arms=['control','shared','both' if (ROOT/'both-result.json').exists() else 'dense']
results={a:json.loads((ROOT/(a+'-result.json')).read_text()) for a in arms}
final=json.loads((ROOT/'final-image.json').read_text())
restored=json.loads((ROOT/'restored.json').read_text())
assert restored['gateway_canary']['passed']
assert not json.loads((ROOT/'cleanup.json').read_text())['errors']
for p in ROOT.iterdir():
    if p.is_file() and p.suffix in ('.json','.py','.sh','.log','.md'):
        shutil.copy2(p,raw/p.name)
engine=ROOT.parent/'glm-native-mtp-20261020/engine'
product=ROOT.parent/'glm-native-mtp-20261020/product'
(raw/'engine.patch').write_bytes(subprocess.check_output(['git','-C',str(engine),'show','--format=fuller','e1ee623547ee8ca85abb4e17091907cb267175fe']))
(raw/'product.patch').write_bytes(subprocess.check_output(['git','-C',str(product),'show','--format=fuller',final['product_commit']]))
lines=['# SparkGLM TP2: corrected FP4 shared and dense groups','',
    '2026-10-20. Focused, uncapped comparison on ennspark01/02. Current corrected-FP4 KDA Spark mode with native MLA fusion is the control. Shared FP4 is tested first; dense FP4 is then tested on top of the accepted configuration. Strict native formats are unchanged.','',
    '| Configuration | Prose tok/s | Code tok/s | Structured tok/s | Quality |','|---|---:|---:|---:|---:|']
for a,x in results.items():
    vals=[x['decode'][n]['decode_tokens_per_second']['median'] for n in ('prose','code','structured')]
    count=sum(q['grade']['status']=='pass' for q in x['quality'])
    lines.append('| '+a+' | '+' | '.join(f'{v:.3f}' for v in vals)+f' | {count}/6 |')
lines+=['','Medians of two repetitions. Every repetition:','', '| Configuration | Workload | Run1 | Run2 |','|---|---|---:|---:|']
for a,x in results.items():
    for name,d in x['decode'].items():
        assert len(d['runs'])==2
        lines.append('| '+a+' | '+name+' | '+' | '.join(f"{row['decode_tokens_per_second']:.3f}" for row in d['runs'])+' |')
lines+=['','## Decision','',f"Selected managed Spark policy: **{selected['arm']}**.",'',
    'The predeclared quick acceptance rule was all six quality cases passing, all timed completion gates passing, at least 2% geometric-mean throughput improvement across the three workloads, and no workload regression greater than 2%. This is a focused qualification threshold, not a statistical confidence claim.','']
for a,d in decisions.items():
    lines.append(f"- {a} versus {d['baseline']}: {(d['geomean_ratio']-1)*100:+.2f}% geometric-mean speed; "+', '.join(f'{n} {(v-1)*100:+.2f}%' for n,v in d['ratios'].items())+f". Quality passed: {d['quality_passed']}. Accepted: {d['accepted']}.")
lines+=['','## Quality and limits','',
    'The same six existing prompts were used for every arm: invoice arithmetic, exact JSON extraction, two Python functions with bounded fixture execution, and the original 70K/113K two-hop retrieval cases. Four short cases ran concurrently; the two long cases ran individually. Greedy decoding, Low reasoning, and 8192 output tokens avoided the historical 24-token truncation problem. Full responses, grades, token usage, request hashes and finish reasons are retained.','',
    'This is a small smoke test. It does not establish broad quality parity, bit-identical outputs, or that RNE alone fixes the old all-group fidelity incident. These switches change both prefill and decode. Shared prefill can use an FP8 cache expanded from the requantized weights; dense prefill also uses the requantized weights.','',
    'Both conversions use the existing corrected GLM weight quantizer, quantize_bf16_to_nvfp4_weight_rne. No quantization kernel or activation quantizer changed. Spark mode remains selected throughout, preserving the corrected DFlash quantizer. MLA and KDA prefill stay native. Each arm uses a fresh isolated prefix/swap cache root.','',
    'Timing uses the same existing RigMark prose/code/structured prompts and the first two saved nonce strings, seed 20260905, temperature 0, top_p 1, Low reasoning and 4096-token allowance. Decode excludes prefill; different generated text and speculative acceptance can influence measured rates. Both ranks use 4x1m, memory utilization 0.91, KV sharding, display carveout and 48 GiB prefix-cache allowance. GPU clock-cap services were disabled throughout timing and restored afterward.','',
    '## Validation and live identity','',
    '- Three targeted Rust mode tests passed, including all shared/dense combinations and Strict rejection of nonnative policies. Product launch tests, converter tests and shell/benchmark-driver checks are recorded in validation.json.',
    '- Changed Rust source passes formatting. Workspace-wide formatting reports a pre-existing wrap in model/types.rs from native-MTP work; it was left untouched. No full benchmark or full engine test campaign was run.',
    f"- Engine: `{final['source']['commit']}`.",f"- Binary: `{final['binary_sha256']}`.",f"- Image: `{final['image_id']}`.",f"- Product commit: `{final['product_commit']}`.",
    '- The final image contains the same tested binary; only accepted product mode defaults were packaged afterward. Both managed ranks have been checked, normal clocks restored, LLooM lease released, and the gateway answered 42. Qwen03/04 was not modified. This is a local unpublished RC.','',
    '- The first managed restore was refused by LLooM memory admission. A normal retry succeeded; no memory budgets, runtime estimates, or container limits were changed. The failure receipt is retained.',
    '- All 201 effective Atlas settings were compared between the accepted experiment and managed serving. Precision and optimization settings match; the managed launch lists the same two RDMA devices in the opposite order. Normal serving restores the clock caps, so the uncapped numbers describe the measurement condition.',
    '- All six control outputs match the preceding MLA-fusion RC in text, reasoning hashes, token counts and finish reasons. The public Spark preset rejects dense-FP4 overrides because that candidate did not qualify.','',
    'Raw receipts and scripts are under raw/. SHA256SUMS covers the bundle.']
(OUT/'RESULT.md').write_text('\n'.join(lines)+'\n')
(OUT/'summary.json').write_text(json.dumps({'selected':selected,'decisions':decisions,'image':final},indent=2)+'\n')
(OUT/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
dest=product/'results/2026-10-20-fp4-groups';assert not dest.exists();shutil.copytree(OUT,dest)
print(OUT/'RESULT.md')
