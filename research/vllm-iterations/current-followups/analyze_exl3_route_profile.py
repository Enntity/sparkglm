#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize EXL3_ROUTE_PROFILE lines emitted by the diagnostic overlay.

The useful cache question is not merely whether each chunk has a large expert.
It is whether the expert that was hot in the *previous* chunk remains hot in the
next chunk.  The previous-chunk columns below model a one- or two-entry online
cache without peeking at the current routing result.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass


LINE_RE = re.compile(
    r"EXL3_ROUTE_PROFILE layer=(?P<layer>\d+) call=(?P<call>\d+) "
    r"tokens=(?P<tokens>\d+) top=(?P<top>[^\s]+)"
)


@dataclass(frozen=True)
class Sample:
    layer: int
    call: int
    tokens: int
    counts: dict[int, int]


def parse(lines: list[str], min_call: int) -> dict[int, list[Sample]]:
    by_layer: dict[int, list[Sample]] = defaultdict(list)
    for line in lines:
        match = LINE_RE.search(line)
        if match is None:
            continue
        call = int(match.group("call"))
        if call < min_call:
            continue
        counts: dict[int, int] = {}
        for item in match.group("top").split(","):
            expert, count = item.split(":", 1)
            counts[int(expert)] = int(count)
        sample = Sample(
            layer=int(match.group("layer")),
            call=call,
            tokens=int(match.group("tokens")),
            counts=counts,
        )
        by_layer[sample.layer].append(sample)
    for samples in by_layer.values():
        samples.sort(key=lambda item: item.call)
    return by_layer


def top_ids(sample: Sample, n: int) -> tuple[int, ...]:
    return tuple(
        expert
        for expert, _ in sorted(
            sample.counts.items(), key=lambda item: (-item[1], item[0])
        )[:n]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-call",
        type=int,
        default=2,
        help="ignore startup/dummy calls before this per-layer call number",
    )
    parser.add_argument(
        "--experts-per-token",
        type=int,
        default=8,
        help="GLM routed experts selected per token",
    )
    args = parser.parse_args()

    by_layer = parse(sys.stdin.readlines(), args.min_call)
    if not by_layer:
        print("no EXL3_ROUTE_PROFILE samples found", file=sys.stderr)
        return 2

    transitions = 0
    same_top1 = 0
    top2_overlap = 0
    current_oracle1_rows = 0
    previous1_rows = 0
    previous2_rows = 0
    transition_rows = 0
    calls = 0
    routed_rows = 0

    print(
        "layer calls top1_repeat prev1_row_pct prev2_row_pct "
        "oracle1_row_pct"
    )
    for layer, samples in sorted(by_layer.items()):
        layer_transitions = 0
        layer_same = 0
        layer_prev1 = 0
        layer_prev2 = 0
        layer_oracle1 = 0
        layer_rows = 0
        previous: Sample | None = None
        for sample in samples:
            if previous is None:
                previous = sample
                continue
            rows = sample.tokens * args.experts_per_token
            prev1 = top_ids(previous, 1)
            prev2 = top_ids(previous, 2)
            curr1 = top_ids(sample, 1)
            curr2 = top_ids(sample, 2)
            hit1 = sum(sample.counts.get(expert, 0) for expert in prev1)
            hit2 = sum(sample.counts.get(expert, 0) for expert in prev2)
            oracle1 = sample.counts[curr1[0]]

            layer_transitions += 1
            layer_same += int(prev1 == curr1)
            top2_overlap += len(set(prev2) & set(curr2))
            layer_prev1 += hit1
            layer_prev2 += hit2
            layer_oracle1 += oracle1
            layer_rows += rows
            transition_rows += rows
            previous1_rows += hit1
            previous2_rows += hit2
            current_oracle1_rows += oracle1
            transitions += 1
            same_top1 += int(prev1 == curr1)
            previous = sample

        calls += len(samples)
        routed_rows += sum(
            sample.tokens * args.experts_per_token for sample in samples
        )
        if layer_transitions:
            print(
                f"{layer:02d} {len(samples):02d} "
                f"{100.0 * layer_same / layer_transitions:7.2f} "
                f"{100.0 * layer_prev1 / layer_rows:7.3f} "
                f"{100.0 * layer_prev2 / layer_rows:7.3f} "
                f"{100.0 * layer_oracle1 / layer_rows:7.3f}"
            )

    print("summary")
    print(f"layers={len(by_layer)} calls={calls} transitions={transitions}")
    print(f"top1_repeat_pct={100.0 * same_top1 / transitions:.3f}")
    print(
        "top2_mean_overlap="
        f"{top2_overlap / transitions:.3f}/2"
    )
    print(
        "previous_chunk_cache1_routed_row_pct="
        f"{100.0 * previous1_rows / transition_rows:.3f}"
    )
    print(
        "previous_chunk_cache2_routed_row_pct="
        f"{100.0 * previous2_rows / transition_rows:.3f}"
    )
    print(
        "current_chunk_oracle1_routed_row_pct="
        f"{100.0 * current_oracle1_rows / transition_rows:.3f}"
    )
    print(
        "profiled_routed_rows="
        f"{routed_rows} transition_routed_rows={transition_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
