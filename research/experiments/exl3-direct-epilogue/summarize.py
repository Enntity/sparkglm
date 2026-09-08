#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize all three paired receipts, never the best sample; stdout is JSON."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(directory):
    arms = {arm: [json.loads((directory / f'{arm}-pair-{pair}.json').read_text())
                  for pair in (1, 2, 3)] for arm in ('reference', 'candidate')}
    cases = []
    active_ratios = []
    for arm, runs in arms.items():
        assert all(run['arm'] == arm for run in runs)
        for key in ('image_id', 'source_sha256', 'extension_sha256', 'harness_sha256'):
            assert len({run[key] for run in runs}) == 1, f'{arm} {key} drift'
    for ref, cand in zip(arms['reference'], arms['candidate']):
        assert ref['harness_sha256'] == cand['harness_sha256']
        assert len(ref['results']) == len(cand['results']) == 14
        assert ref['image_id'] != cand['image_id']
    for index, first in enumerate(arms['reference'][0]['results']):
        ratios = []
        for ref, cand in zip(arms['reference'], arms['candidate']):
            base, new = ref['results'][index], cand['results'][index]
            for key in ('counts', 'cap', 'seed', 'tokens', 'iterations', 'repeats', 'warmup'):
                assert base[key] == new[key], f'pair mismatch: {key}'
            assert base['counts'] == first['counts']
            for result in (base, new):
                assert result['graph_replay_checked'] and result['red_zones_checked']
                assert result['median_ms'] > 0
            ratios.append(new['median_ms'] / base['median_ms'])
        if max(first['counts']) > first['cap']:
            active_ratios.extend(ratios)
        cases.append({'counts': first['counts'], 'paired_latency_ratios': ratios,
                      'geomean_latency_change_percent': 100 * (statistics.geometric_mean(ratios) - 1)})
    change = 100 * (statistics.geometric_mean(active_ratios) - 1)
    return {'schema': 'sparkglm.direct-epilogue-summary/v1',
            'scope': 'synthetic model-free grouped operator; not endpoint throughput',
            'active_geomean_latency_change_percent': change,
            'performance_screen_passed': change <= -3 and all(
                not all(r > 1.02 for r in case['paired_latency_ratios'])
                for case in cases if max(case['counts']) > 128),
            'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2, allow_nan=False))
