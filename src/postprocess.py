"""postprocess.py — единый финальный постпроцессинг сегментов Part A.

Правила (rules.py, obstacle_fire.py) выдают "сырые" сегменты; здесь они
приводятся к виду, который лучше матчится по tIoU:
  1. обрезка по [0, duration];
  2. склейка фрагментов одного класса с разрывом <= merge_gap
     (FAQ: одновременные события одного класса -> один сегмент);
  3. выброс коротышей < min_dur.

Параметры — ПО КЛАССУ, потому что у классов разная природа длительности
(jaywalking рвётся на окклюзиях, congestion длится десятки секунд, а
red_light — 2-4 с, и склейка соседних проездов разных машин с gap=3 с
убила бы IoU). Подбирать на своей разметке: tools/dev_loop.py.
"""
from __future__ import annotations

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
    "curb_mount":          (1.0, 0.3),   # диагностика, не отправляется
}
DEFAULT_POST = (0.5, 0.3)


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
        out += [[round(s, 2), round(e, 2), lbl] for s, e in merged if e - s >= min_dur]
    return sorted(out)
