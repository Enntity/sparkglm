"""SPDX-License-Identifier: AGPL-3.0-only"""

import math


def _check_number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be int or float" % what)
    try:
        f = float(value)
    except (OverflowError, ValueError):
        raise ValueError("%s not representable as finite float" % what)
    if not math.isfinite(f):
        raise ValueError("%s must be finite" % what)
    return f


def _validate_common(d, keys, label):
    if not isinstance(d, dict):
        raise ValueError("%s must be a dict" % label)
    if set(d.keys()) != keys:
        raise ValueError("%s must have exactly keys %s" % (label, sorted(keys)))


def summarize_events(events, windows):
    if not isinstance(events, list) or not isinstance(windows, list):
        raise ValueError("events and windows must be lists")

    parsed_events = []
    for ev in events:
        _validate_common(ev, {"name", "start", "end"}, "event")
        name = ev["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("event name must be a nonempty string")
        start = _check_number(ev["start"], "event start")
        end = _check_number(ev["end"], "event end")
        if end < start:
            raise ValueError("event requires end >= start")
        parsed_events.append((name, start, end))

    parsed_windows = []
    for w in windows:
        _validate_common(w, {"id", "start", "end"}, "window")
        wid = w["id"]
        if not isinstance(wid, str) or not wid:
            raise ValueError("window id must be a nonempty string")
        start = _check_number(w["start"], "window start")
        end = _check_number(w["end"], "window end")
        if end <= start:
            raise ValueError("window requires end > start")
        parsed_windows.append((wid, start, end))

    parsed_windows.sort(key=lambda t: (t[1], t[2], t[0]))
    for i in range(1, len(parsed_windows)):
        if parsed_windows[i][1] < parsed_windows[i - 1][2]:
            raise ValueError("windows must be pairwise disjoint")

    out_windows = []
    for wid, wstart, wend in parsed_windows:
        per_name = {}
        clipped = []
        for name, estart, eend in parsed_events:
            cs = estart if estart > wstart else wstart
            ce = eend if eend < wend else wend
            if cs < ce:
                clipped.append((cs, ce))
                d = ce - cs
                ent = per_name.get(name)
                if ent is None:
                    per_name[name] = [1, [d]]
                else:
                    ent[0] += 1
                    ent[1].append(d)

        covered = 0.0
        if clipped:
            clipped.sort()
            merged_s, merged_e = clipped[0]
            parts = []
            for cs, ce in clipped[1:]:
                if cs <= merged_e:
                    if ce > merged_e:
                        merged_e = ce
                else:
                    parts.append(merged_e - merged_s)
                    merged_s, merged_e = cs, ce
            parts.append(merged_e - merged_s)
            covered = math.fsum(parts)
        if not math.isfinite(covered):
            raise ValueError("covered_duration not finite")

        duration = wend - wstart
        if not math.isfinite(duration):
            raise ValueError("duration not finite")
        uncovered = math.fsum([duration, -covered])
        if not math.isfinite(uncovered):
            raise ValueError("uncovered_duration not finite")

        by_name = [
            {
                "name": name,
                "count": per_name[name][0],
                "duration": math.fsum(per_name[name][1]),
            }
            for name in sorted(per_name)
        ]
        for e in by_name:
            if not math.isfinite(e["duration"]):
                raise ValueError("duration not finite")

        out_windows.append(
            {
                "id": wid,
                "start": wstart,
                "end": wend,
                "duration": duration,
                "covered_duration": covered,
                "uncovered_duration": uncovered,
                "by_name": by_name,
            }
        )

    return {"windows": out_windows}
