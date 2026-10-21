# SPDX-License-Identifier: AGPL-3.0-only
import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
out=r.parents[1]/'outputs/glm-mla-fusion-20261020'
raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
comparison=json.loads((r/'comparison-summary.json').read_text())
result=json.loads((r/'result.json').read_text())
parity=json.loads((r/'parity-summary.json').read_text())
identity=json.loads((r/'candidate-image.json').read_text())
restored=json.loads((r/'restored.json').read_text())
assert restored['gateway_canary']['passed'] and all(parity['exact'].values())
assert not json.loads((r/'cleanup.json').read_text())['errors']
for w,d in result['decode'].items():
 assert len(d['runs'])==3 and d['completion_gate']['passed']==3
 for x in d['runs']:
  assert x['finish_reason']=='stop' and x['stream_done_marker']
  assert abs((x['completion_tokens']-1)/x['decode_seconds']-x['decode_tokens_per_second'])<.002
for p in r.iterdir():
 if p.suffix in ['.json','.py','.log','.sh'] and p.is_file() and p.name not in ['edit_tests.py','implement.py','finalize_source.py','prepare_run.py']:
  shutil.copy2(p,raw/p.name)
shutil.copy2(r.parent/'glm-spark-nvfp4-quick-20261020/result.json',raw/'previous-spark-result.json')
(raw/'engine.patch').write_bytes(subprocess.check_output(['git','-C',str(r/'engine'),'show','--format=fuller','HEAD']))
clock=json.loads((r/'uncapped.json').read_text())
assert all(x['cap_service']=='inactive' for x in clock)
exact9=all(row['exact'] for d in comparison.values() for row in d['output_matches_previous'])
lines=['# SparkGLM TP2: native FP8 MLA fusion','',
 '2026-10-20. Restored the existing MLA-front fusion for native FP8 block weights on TP2. The existing switch stays on in the shipped capacity profile. No weight conversion, FP4 rounding policy, drafter policy, or serving limit changed. Strict native-MTP work is retained.','',
 '| Workload | Previous Spark tok/s | Fused Spark tok/s | Gain | Saved NVFP4 tok/s | Versus NVFP4 |',
 '|---|---:|---:|---:|---:|---:|']
for name,x in comparison.items():lines.append(f"| {name.capitalize()} | {x['before']:.3f} | {x['after']:.3f} | {x['gain_pct']:+.2f}% | {x['nvfp4']:.3f} | {x['vs_nvfp4_pct']:+.2f}% |")
lines+=['','Medians of three repetitions, both ranks uncapped. Every measured repetition:','',
 '| Workload | Run 1 | Run 2 | Run 3 |','|---|---:|---:|---:|']
for name,x in comparison.items():lines.append('| '+name.capitalize()+' | '+' | '.join(f'{n:.3f}' for n in x['repetitions'])+' |')
lines+=['','## Exactness and focused validation','',
 '- GPU tests passed with PDL enabled: native raw projections before normalization, with and without prefetch; the native full front including normalized latents, index data and FP8 KV-cache bytes; and the original MXFP8 front. All widths 1–32, four start positions, unsharded and both TP2 ownership cases. The full-front checks cover 384 cases per format. Native scales include non-power-of-two FP32 values.',
 f"- Five served fusion-off/on checks passed: arithmetic, JSON, prose, code and structured output; all {parity['tokens']} token IDs, response text, reasoning text and finish reasons matched exactly.",
 f"- Nine timed outputs match the previous Spark receipt in text hash, reasoning hash, token counts and finish reason: {exact9}. Per-request details are in raw/comparison-summary.json.",
 '- Two targeted Rust CPU tests and 27 product installation tests passed. Three GPU tests passed. The first GPU attempt failed before MLA because the builder exposed link-time CUDA stubs; rerunning with the actual driver libraries passed. Both attempts are retained.',
 '- All nine timed completions stopped naturally and passed the existing completion gates, including JSON checks. No broader quality, context, concurrency, or prefill campaign was run.','',
 '## What changed','',
 'The fused front now accepts the checkpoint’s native FP8 q_a/kv_a twins and reuses the exact existing FP8-block CTA arithmetic. It preserves the scale application, accumulation order, BF16 rounding, and fused-mid cache/norm code. Native scale-prefetch lengths use FP32 128x128 block sizes. Native activation is limited to TP2 and excludes packed prefill; unsupported layouts keep the existing path.',
 '', '## Measurement and provenance','',
 'The same RigMark prompts, nonce strings, temperature 0, top_p 1, low reasoning, seed 20260905, 4,096-token allowance and streaming protocol were used. Decode rate is (completion_tokens - 1)/(last output time - first output time), excluding prefill. Effective ATLAS environment settings exactly match the previous Spark timing run. Each run uses FP4-Local, Spark mode, DFlash2, 4x1m, memory utilization 0.91, KV sharding, display carveout and 48 GiB disk prefix caching.',
 '',
 'The NVFP4 column reuses the best saved three-run median per workload: October 12 RC4 for code, October 15 for prose and structured. The older checkpoint/build generated different outputs; code used a 4x512k capacity profile. These are historical same-pair comparisons, not a new old-build A/B or a universal performance guarantee.',
 '',f"Nine-request wrapper including its warmup: {result['finished_at']-result['started_at']:.1f} seconds.",
 '',f"Engine commit: `{identity['source']['commit']}`.",f"Image: `{identity['image_id']}`.",f"Image tag: `{identity['tag']}`.",f"Binary SHA256: `{identity['binary_sha256']}`.",
 '', '## Pair state','',
 ('The validated RC is installed as managed Spark serving on both 01/02 nodes. The normal clock caps are restored; the reservation is released and the gateway canary returned 42. This is a local unpublished RC.' if restored['identities'][0]['image']==identity['image_id'] else 'The previous managed Spark image is restored on 01/02 with normal clock caps; the reservation is released and the gateway canary returned 42. The new local RC is saved on both nodes but was not promoted.'),
 '', 'Raw requests, completions, comparison data, GPU tests, build identity, launch settings, and restoration evidence are under raw/. SHA256SUMS covers every artifact.']
(out/'RESULT.md').write_text('\n'.join(lines)+'\n')
(out/'summary.json').write_text(json.dumps({'comparison':comparison,'exact_short_tokens':parity['tokens'],'timed_outputs_match_previous':exact9,'image':identity,'restored':restored['identities']},indent=2)+'\n')
(out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
dest=r/'product/results/2026-10-20-native-mla-fusion'
assert not dest.exists();shutil.copytree(out,dest)
print(out/'RESULT.md')
