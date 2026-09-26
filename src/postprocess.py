"""postprocess.py — the single final post-processing of Part A segments.

The rules (rules.py, obstacle_fire.py) emit "raw" segments; here they are
brought to a form that matches better under tIoU:
  1. clipping to [0, duration];
  2. merging fragments of the same class with a gap <= merge_gap
     (FAQ: simultaneous events of one class -> one segment);
  3. dropping short ones < min_dur.

Parameters are PER CLASS because classes differ in the nature of their duration
(jaywalking breaks up on occlusions, congestion lasts tens of seconds, while
red_light lasts 2-4 s, and merging adjacent passes of different cars with gap=3 s
would kill the IoU). Tune on our own labels: tools/dev_loop.py.
"""
from __future__ import annotations

import math

# class -> (merge_gap_sec, min_duration_sec)
CLASS_POST: dict[str, tuple[float, float]] = {
    "congestion":          (3.0, 5.0),
    "stopped_vehicle":     (2.0, 10.0),
    "jaywalking":          (1.5, 1.0),
    "red_light":           (0.3, 0.5),
    "stop_line":           (1.0, 1.0),
    "wrong_way":           (1.0, 1.0),
    "illegal_u_turn":      (1.0, 1.0),
    "illegal_turn":        (1.0, 1.0),
    "solid_line_crossing": (0.5, 0.5),
    "failure_to_yield":    (0.5, 0.3),
    "accident":            (2.0, 1.0),
    "near_miss":           (1.0, 0.5),
    "road_obstacle":       (2.0, 4.0),
    "fire_smoke":          (3.0, 2.0),
    "curb_mount":          (1.0, 0.3),   # diagnostic, not submitted
}
DEFAULT_POST = (0.5, 0.3)


EDGE_SNAP_SEC = 1.0   # an event closer than this to the clip edge continues past it: 0 / duration (as in the labels)


def postprocess(events, duration: float | None = None, classes=None,
                params: dict | None = None) -> list[list]:
    params = {**CLASS_POST, **(params or {})}
    by_label: dict[str, list[list[float]]] = {}
    for s, e, lbl in events:
        if classes is not None and lbl not in classes:
            continue
        s, e = float(s), float(e)
        if duration is not None:
            s, e = max(0.0, s), min(e, duration)
        if e <= s:
            continue
        by_label.setdefault(lbl, []).append([s, e])

    out = []
    for lbl, ivs in by_label.items():
        gap, min_dur = params.get(lbl, DEFAULT_POST)
        ivs.sort()
        merged = [ivs[0][:]]
        for s, e in ivs[1:]:
            if s - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            if e - s < min_dur:
                continue
            s = 0.0 if s <= EDGE_SNAP_SEC else round(s, 2)
            if duration is not None and e >= duration - EDGE_SNAP_SEC:
                e = math.floor(duration * 100) / 100     # round down: must not exceed duration after rounding
            else:
                e = round(e, 2)
            out.append([s, e, lbl])
    return sorted(out)
