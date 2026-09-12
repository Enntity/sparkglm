# SPDX-License-Identifier: AGPL-3.0-only
import math


def _is_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return True
    return False


def _to_finite_float(value, label):
    if not _is_number(value):
        raise ValueError("invalid type for %s" % label)
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("value not representable as float for %s" % label)
    if not math.isfinite(result):
        raise ValueError("nonfinite value for %s" % label)
    return result


def _check_keys_exact(obj, expected_keys, label):
    if type(obj) is not dict:
        raise ValueError("invalid type for %s" % label)
    if set(obj.keys()) != set(expected_keys):
        raise ValueError("invalid keys for %s" % label)


def _check_nonempty_str(value, label):
    if not isinstance(value, str):
        raise ValueError("invalid type for %s" % label)
    if len(value) == 0:
        raise ValueError("empty string for %s" % label)


def _validate_events(events):
    validated = []
    for index, event in enumerate(events):
        label = "event[%d]" % index
        _check_keys_exact(event, ("name", "start", "end"), label)
        _check_nonempty_str(event["name"], label + ".name")
        start = _to_finite_float(event["start"], label + ".start")
        end = _to_finite_float(event["end"], label + ".end")
        if end < start:
            raise ValueError("negative event duration for %s" % label)
        validated.append((event["name"], start, end))
    return validated


def _validate_windows(windows):
    validated = []
    seen_ids = set()
    for index, window in enumerate(windows):
        label = "window[%d]" % index
        _check_keys_exact(window, ("id", "start", "end"), label)
        _check_nonempty_str(window["id"], label + ".id")
        window_id = window["id"]
        if window_id in seen_ids:
            raise ValueError("duplicate window id for %s" % label)
        seen_ids.add(window_id)
        start = _to_finite_float(window["start"], label + ".start")
        end = _to_finite_float(window["end"], label + ".end")
        if not (end > start):
            raise ValueError("non-positive window duration for %s" % label)
        validated.append((window_id, start, end))
    validated.sort(key=lambda item: (item[1], item[2], item[0]))
    for i in range(1, len(validated)):
        prev = validated[i - 1]
        cur = validated[i]
        if cur[1] < prev[2]:
            raise ValueError("overlapping windows")
    return validated


def _union_length(intervals):
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    total_pieces = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total_pieces.append(current_end - current_start)
            current_start, current_end = start, end
        else:
            if end > current_end:
                current_end = end
    total_pieces.append(current_end - current_start)
    return math.fsum(total_pieces)


def summarize_events(events, windows):
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    if not isinstance(windows, list):
        raise ValueError("windows must be a list")

    validated_events = _validate_events(events)
    validated_windows = _validate_windows(windows)

    output_windows = []
    for window_id, win_start, win_end in validated_windows:
        window_duration = win_end - win_start
        name_durations = {}
        name_counts = {}
        clipped_intervals = []
        for name, ev_start, ev_end in validated_events:
            clip_start = ev_start if ev_start > win_start else win_start
            clip_end = ev_end if ev_end < win_end else win_end
            clipped = clip_end - clip_start
            if clipped > 0.0:
                clipped_intervals.append((clip_start, clip_end))
                contributions = name_durations.get(name)
                if contributions is None:
                    name_durations[name] = [clipped]
                    name_counts[name] = 1
                else:
                    contributions.append(clipped)
                    name_counts[name] = name_counts[name] + 1

        covered = _union_length(clipped_intervals)
        uncovered = window_duration - covered

        by_name = []
        for name in sorted(name_durations.keys()):
            duration = math.fsum(name_durations[name])
            if not math.isfinite(duration):
                raise ValueError("nonfinite aggregate duration")
            by_name.append({
                "name": name,
                "count": name_counts[name],
                "duration": duration,
            })

        if not math.isfinite(covered):
            raise ValueError("nonfinite covered_duration")
        if not math.isfinite(uncovered):
            raise ValueError("nonfinite uncovered_duration")
        if not math.isfinite(window_duration):
            raise ValueError("nonfinite duration")

        output_windows.append({
            "id": window_id,
            "start": win_start,
            "end": win_end,
            "duration": window_duration,
            "covered_duration": covered,
            "uncovered_duration": uncovered,
            "by_name": by_name,
        })

    return {"windows": output_windows}
