#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Local metadata/plan checks only; no compiler, CUDA, model or network calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
SOURCE = P.parent / 'moe-tp-shape/fixture.hpp'


def retained_routes(skew, owner):
    # Exact ID formula from the SHA-pinned prior fixture. Weights/inputs will
    # be reused from its C++ fixture rather than reinvented in the GPU screen.
    values = []
    for t in range(3):
        for j in range(8):
            half = int(j == 7) if skew else j & 1
            if skew and owner & 1:
                half ^= 1
            local = (owner * 13 + j * 11) % 144 if skew else (owner * 29 + t * 19 + (j // 2) * 31) % 144
            values.append(half * 144 + local)
    return values


def qualify(values):
    assert len(values) == 96 and all(0 <= e < 288 for e in values)
    for token in range(12):
        assert len(set(values[token * 8:(token + 1) * 8])) == 8
    counts = Counter(values)
    assert max(counts.values()) <= 12
    for owner in range(4):
        for temporal in range(3):
            token = owner * 3 + temporal
            assert divmod(token, 3) == (owner, temporal)
            assert token * 2048 + 2048 <= 12 * 2048
            assert 12 * 2048 + token * 256 + 256 <= 12 * 2304
    assert 16 + 96 * 16 * 8 == 12304 <= 16384
    assert 96 * 2048 // 2 == 98304 and 96 * 2048 // 16 == 12288
    return counts


def main():
    cases = []
    for skew in (False, True):
        owners = [retained_routes(skew, owner) for owner in range(4)]
        values = sum(owners, [])
        joint = qualify(values)
        serial = [Counter(v) for v in owners]
        visits = [sum(sum(e // 144 == rank for e in c) for c in serial) for rank in (0, 1)]
        unique = [sum(e // 144 == rank for e in joint) for rank in (0, 1)]
        assert visits == unique  # no cross-owner weight reuse in retained cases
        assert len(joint) == (32 if skew else 96)
        cases.append(dict(skew=skew, owner_unique=[len(c) for c in serial], joint_unique=len(joint),
                          max_rows_per_expert=max(joint.values()), serial_rank_expert_visits=visits,
                          joint_rank_unique=unique, cross_owner_overlap=False))
    shared_ids = [0, 143, 144, 287, 1, 142, 145, 286] * 12
    shared = qualify(shared_ids)
    assert len(shared) == 8 and all(v == 12 for v in shared.values())
    malformed = shared_ids.copy(); malformed[1] = malformed[0]
    try:
        qualify(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate top8 expert was admitted')
    print(json.dumps(dict(status='LOCAL_PLAN_METADATA_CHECKS_PASS', cuda_build=False, gpu_run=False,
                          fixture_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                          primary=cases[0], secondary=cases[1],
                          correctness_only_shared_fixture=dict(experts=8, rows_per_expert=12),
                          fixed_routed_pipeline_speedup_gate=1.3,
                          future_overlap_sweep_percent=[0, 25, 50, 100], future_sweep_authorized=False), indent=2))


if __name__ == '__main__':
    main()
