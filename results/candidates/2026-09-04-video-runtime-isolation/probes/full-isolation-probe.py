# SPDX-License-Identifier: Apache-2.0
"""Retain exact video replays plus speculative-work counters; no model changes."""
import argparse, json, re, subprocess, threading, time, urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--arm', required=True)
p.add_argument('--engine-commit', required=True)
p.add_argument('--runs', type=int, default=5)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
base = 'http://127.0.0.1:8890'

def metrics():
    with urllib.request.urlopen(base + '/metrics', timeout=15) as response:
        text = response.read().decode()
    selected = {}
    for line in text.splitlines():
        if line.startswith('#') or not line: continue
        if not any(k in line for k in ['spec_decode', 'generation_tokens', 'prompt_tokens', 'preemptions', 'prefix_cache', 'request_success']): continue
        match = re.match(r'^(\S+)\s+([-+0-9.eE]+)(?:\s|$)', line)
        if match: selected[match[1]] = float(match[2])
    return selected

for run in range(a.runs + 1):
    name = 'warmup' if run == 0 else 'r' + str(run)
    prefix = a.output / (a.arm + '-' + name)
    if Path(str(prefix) + '.json').exists():
        raise RuntimeError('Refusing to overwrite a retained replay: ' + str(prefix))
    time.sleep(12)  # let the previous engine stats interval flush; outside timing
    before = metrics()
    gpu = []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            value = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            gpu.append({'monotonic': time.monotonic(), 'csv': value.stdout.strip(), 'returncode': value.returncode})
            stop.wait(1)
    thread = threading.Thread(target=sample)
    thread.start()
    command = ['python3', str(a.root / 'benchmarks/four_stream_video.py'), 'capture', '--base-url', base, '--model', 'glm-5.3-flash-exl3', '--streams', '4', '--max-tokens', '400', '--prompt-style', 'field-guide', '--prompt-tokens', '16000', '--prompt-salt', 'current-best-video-20260903-r1', '--cache-salt', 'isolation-' + a.arm + '-' + name + '-' + str(time.time_ns()), '--stagger-ms', '1000', '--recipe-label', 'diagnostic-' + a.arm, '--engine-commit', a.engine_commit, '--output', str(prefix) + '.json']
    try:
        subprocess.run(command, check=True)
    finally:
        stop.set()
        thread.join()
    time.sleep(12)  # metrics are batched; this delay is excluded from replay wall time
    after = metrics()
    delta = {k: after[k] - before.get(k, 0) for k in after if '_total' in k}
    receipt = {'arm': a.arm, 'run': name, 'metrics_before': before, 'metrics_after': after, 'counter_deltas': delta, 'gpu_samples': gpu}
    Path(str(prefix) + '-work.json').write_text(json.dumps(receipt, indent=2) + '\n')
    replay = json.loads(Path(str(prefix) + '.json').read_text())
    if len(replay['requests']) != 4 or any(r['error'] or r['completion_tokens'] != 400 for r in replay['requests']):
        raise RuntimeError('Incomplete replay retained; stopping campaign')
    print(json.dumps({'arm':a.arm,'run':name,'deltas':delta}), flush=True)
