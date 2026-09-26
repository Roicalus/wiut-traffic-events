"""Синтетические проверки правил и Part B без YOLO/GPU.

    python -m pytest tests -q
"""
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import rules  # noqa: E402
from src.postprocess import postprocess  # noqa: E402
from src.stitch_tracks import stitch_records  # noqa: E402

FPS, STRIDE = 25.0, 3
ZONES = rules.load_zones(ROOT / "zones.json")


def point_in(zone, exclude=(), seed=0):
    rng = random.Random(seed)
    poly = ZONES[zone]
    xs, ys = poly[:, 0], poly[:, 1]
    for _ in range(20000):
        p = (rng.uniform(xs.min(), xs.max()), rng.uniform(ys.min(), ys.max()))
        if rules.in_zone(p, poly) and not any(rules.in_zone(p, ZONES[z]) for z in exclude):
            return p
    raise RuntimeError(zone)


def track(tid, cls, path, t0, t1, w=120, h=80, jitter=0.0, seed=0):
    """path(t) -> (x, y) опорной (нижней центральной) точки."""
    rng = np.random.default_rng(seed)
    out = []
    f0, f1 = int(t0 * FPS), int(t1 * FPS)
    for f in range(f0 - f0 % STRIDE, f1, STRIDE):
        t = f / FPS
        x, y = path(t)
        x += rng.normal(0, jitter)
        y += rng.normal(0, jitter)
        out.append({"frame": f, "t_sec": round(t, 3), "track_id": tid, "cls": cls,
                    "x1": x - w / 2, "y1": y - h, "x2": x + w / 2, "y2": y})
    return out


def events_of(records, label):
    stitch_records(records)
    return [e for e in rules.compute_events(records, ZONES, None) if e[2] == label]


def test_stationary_jitter_is_still():
    recs = track(1, 2, lambda t: (3000, 1600), 0, 10, w=50, h=35, jitter=2.0)
    s = rules.annotate(recs, ZONES)
    assert np.median([x["speed"] for x in s[5:]]) < 0.15


def test_stopped_vehicle_single_car():
    p = point_in("crossroad")
    ev = events_of(track(1, 2, lambda t: p, 0, 25, jitter=1.5), "stopped_vehicle")
    assert len(ev) == 1 and ev[0][1] - ev[0][0] > 20


def _queue(n, t_end, seed0=10):
    recs = []
    for k in range(n):
        p = point_in("queue_zone", seed=k + seed0)
        recs += track(k + 1, 2, lambda t, p=p: p, 0, t_end, jitter=1.0, seed=k)
    stitch_records(recs)
    return recs


def test_queue_on_red_is_neither_congestion_nor_stopped():
    """Очередь на красный разъезжается на зелёном — это не затор (раньше
    congestion было каждую фазу красного) и не stopped_vehicle."""
    light = [[round(t, 2), "red"] for t in np.arange(0, 30, 0.1)]
    ev = rules.compute_events(_queue(6, 30), ZONES, light)
    assert not any(e[2] in ("congestion", "stopped_vehicle") for e in ev)


def test_queue_standing_through_green_is_congestion():
    light = [[round(t, 2), "red" if t < 20 else "green"] for t in np.arange(0, 50, 0.1)]
    ev = rules.compute_events(_queue(6, 50), ZONES, light)
    assert any(e[2] == "congestion" for e in ev)
    assert not any(e[2] == "stopped_vehicle" for e in ev)


PED_SAFE = ("crossing_far", "crossing_near", "sidewalk_island_1", "sidewalk_island_2")


def test_jaywalking():
    p = point_in("crossroad", exclude=PED_SAFE, seed=3)
    ev = events_of(track(1, 0, lambda t: (p[0] + 20 * t, p[1]), 0, 5, w=30, h=80), "jaywalking")
    assert len(ev) == 1


def test_pedestrian_on_island_is_not_jaywalking():
    """Островок с плиткой внутри полигона crossroad: люди идут по нему с
    одной зебры на другую (C3897 — большинство ложных jaywalking)."""
    p = point_in("sidewalk_island_1", seed=4)
    ev = events_of(track(1, 0, lambda t: (p[0] + 5 * t, p[1]), 0, 5, w=30, h=80), "jaywalking")
    assert ev == []


def test_lone_stopped_car_during_queue_elsewhere():
    """Одна машина 30 с стоит посреди площади, а в очереди на магистрали
    в это же время стоят 6 машин: congestion — там, stopped_vehicle — здесь
    (C3897, 210-246 с: раньше терялось из-за пересечения с congestion)."""
    recs = []
    for k in range(6):
        q = point_in("queue_zone", seed=k + 10)
        recs += track(k + 1, 2, lambda t, q=q: q, 0, 30, jitter=1.0, seed=k)
    p = point_in("crossroad", exclude=PED_SAFE, seed=21)
    recs += track(50, 2, lambda t: p, 0, 30, jitter=1.0, seed=50)
    stopped = events_of(recs, "stopped_vehicle")
    assert len(stopped) == 1 and stopped[0][1] - stopped[0][0] > 25


def test_queue_neighbours_are_not_accident():
    p = point_in("queue_zone", seed=5)
    recs = track(1, 2, lambda t: p, 0, 20, jitter=1.0, seed=1)
    recs += track(2, 2, lambda t: (p[0] + 20, p[1] + 10), 0, 20, jitter=1.0, seed=2)
    assert events_of(recs, "accident") == []


def test_collision_then_stop_is_accident():
    p = point_in("crossroad", seed=7)
    a = lambda t: (p[0] - 300 + 60 * min(t, 5), p[1])   # едет 5 с, потом стоит
    b = lambda t: (p[0] + 300 - 60 * min(t, 5), p[1])
    recs = track(1, 2, a, 0, 12, seed=1) + track(2, 2, b, 0, 12, seed=2)
    ev = events_of(recs, "accident")
    assert len(ev) == 1 and 3.5 <= ev[0][0] <= 5.5


def test_solid_line_hysteresis():
    (x1, y1), (x2, y2) = ZONES["solid_line"]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ux, uy = (x2 - x1), (y2 - y1)
    n = np.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    nx, ny = -uy, ux
    along = lambda t: (mx + ux * 40 * t, my + uy * 40 * t)   # вдоль линии, дрожит поперёк
    recs = track(1, 2, along, 0, 10, jitter=4.0)
    assert events_of(recs, "solid_line_crossing") == []
    across = lambda t: (mx + nx * (-150 + 60 * t), my + ny * (-150 + 60 * t))
    recs = track(2, 2, across, 0, 6, jitter=2.0)
    assert len(events_of(recs, "solid_line_crossing")) == 1


def test_candidate_pairs_match_bruteforce():
    rng = np.random.default_rng(0)
    objs = {}
    for k in range(40):
        x0, y0 = rng.uniform(0, 3000, 2)
        vx, vy = rng.uniform(-80, 80, 2)
        t0 = rng.uniform(0, 20)
        recs = track(k, 2 if k % 3 else 0, lambda t, x0=x0, y0=y0, vx=vx, vy=vy, t0=t0:
                     (x0 + vx * (t - t0), y0 + vy * (t - t0)), t0, t0 + rng.uniform(2, 10))
        objs[("p" if k % 3 == 0 else "v", k)] = rules.annotate(recs, {})
    fast = set(rules._candidate_pairs(objs))
    keys = sorted(objs)
    brute = set()
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == "p" and b[0] == "p":
                continue
            ap = rules._closest_approach(objs[a], objs[b])
            if ap and min(d for _, d, _ in ap) < rules.PAIR_PREFILTER_NORM:
                brute.add((a, b))
    assert brute <= fast


def test_postprocess():
    ev = [[0, 3, "jaywalking"], [4, 6, "jaywalking"], [10, 10.5, "jaywalking"],
          [5, 8, "red_light"], [8.5, 11, "red_light"], [-1, 99, "congestion"]]
    out = postprocess(ev, duration=60)
    assert [0, 6, "jaywalking"] in out
    assert [10, 10.5, "jaywalking"] not in out
    assert sum(1 for e in out if e[2] == "red_light") == 2
    assert [0, 60, "congestion"] in out


def _light(lit, tint=(0, 60, 140), w=60, h=150):
    """Корпус светофора с одной горящей секцией и "закатным" фоном."""
    import cv2
    img = np.full((h, w, 3), tint, np.uint8)
    cv2.rectangle(img, (8, 4), (w - 8, h - 4), (30, 30, 30), -1)
    colors = {"red": (40, 40, 255), "yellow": (40, 220, 255), "green": (200, 255, 60)}
    for k, name in enumerate(("red", "yellow", "green")):
        cy = int(h * (k + 0.5) / 3)
        cv2.circle(img, (w // 2, cy), 14, colors[name] if name == lit else (45, 45, 45), -1)
    return img


def test_light_position_classifier():
    from src.light_state import classify_roi
    for lit in ("red", "yellow", "green"):
        assert classify_roi(_light(lit), mode="position") == lit
    dark = _light(None)
    assert classify_roi(dark, mode="position") == "unknown"


def test_wrong_way_only_in_one_way_zones():
    road = point_in("roadway", seed=11)
    plaza = point_in("crossroad", exclude=("crossing_far", "crossing_near"), seed=12)
    recs = []
    for k in range(12):                       # основной поток по roadway
        recs += track(k, 2, lambda t, k=k: (road[0] - 60 * t, road[1] - 16 * t), k, k + 6, seed=k)
    for k in range(12):                       # на площади — два встречных потока, это норма
        sgn = 1 if k % 2 else -1
        recs += track(100 + k, 2, lambda t, k=k, sgn=sgn: (plaza[0] + sgn * 70 * (t - k), plaza[1]),
                      k, k + 4, seed=k)
    recs += track(999, 2, lambda t: (road[0] + 60 * (t - 3), road[1] + 16 * (t - 3)), 3, 9, seed=3)
    stitch_records(recs)
    ids = rules.detect_wrong_way(
        {k: rules.annotate(v, ZONES) for k, v in rules.group_by_object(recs).items()},
        {"roadway", "roadway_before_queue", "crossroad"}, return_ids=True)[1]
    assert ids == [999], ids


def test_front_of_queue_on_red_is_not_stopped_vehicle():
    p = point_in("stop_line", seed=21)
    move_at = 40.0
    path = lambda t: p if t < move_at else (p[0], p[1] + 80 * (t - move_at))
    recs = track(1, 5, path, 0, 45, w=300, h=200)
    light = [[round(t, 2), "red" if t < move_at - 1 else "green"] for t in np.arange(0, 45, 0.12)]
    stitch_records(recs)
    ev = [e for e in rules.compute_events(recs, ZONES, light) if e[2] == "stopped_vehicle"]
    assert ev == []
    # та же стоянка посреди площади, без светофора рядом — событие
    q = point_in("crossroad", exclude=("crossing_far", "crossing_near", "stop_line"), seed=22)
    recs = track(2, 2, lambda t: q, 0, 25)
    stitch_records(recs)
    assert len([e for e in rules.compute_events(recs, ZONES, light) if e[2] == "stopped_vehicle"]) == 1


def test_green_to_red_without_yellow_needs_to_hold():
    """Машина закрыла зелёную секцию на 2.7 с, негорящая красная линза днём
    читается как горящая (C3896, 190 с) — это не переключение на красный."""
    from src.light_state import smooth_states
    raw = [(t / 10, "green") for t in range(0, 100)]
    raw += [(t / 10, "red") for t in range(100, 127)]          # 2.7 с "красного"
    raw += [(t / 10, "green") for t in range(127, 200)]
    assert {s for t, s in smooth_states(raw) if t >= 0.5} == {"green"}
    # настоящий цикл: зелёный -> жёлтый -> красный переключается сразу
    raw = [(t / 10, "green") for t in range(0, 50)] + [(t / 10, "yellow") for t in range(50, 80)]
    raw += [(t / 10, "red") for t in range(80, 120)]
    states = dict(smooth_states(raw))
    assert states[8.2] == "red"


def test_car_driving_over_island_is_curb_mount():
    """Заехал на тротуарный островок (C3905, ~100 с): диагностическое
    curb_mount, не illegal_turn (своих классов в сабмите быть не может)."""
    a, b = ZONES["sidewalk_island_2"][0], ZONES["sidewalk_island_2"][2]
    path = lambda t: (a[0] + (b[0] - a[0]) * t / 3, a[1] + (b[1] - a[1]) * t / 3)
    recs = track(1, 2, path, 0, 3, w=300, h=200)
    stitch_records(recs)
    ev = rules.compute_events_debug(recs, ZONES, None)
    assert [e[2] for e in ev if e[2] in ("curb_mount", "illegal_turn")] == ["curb_mount"]


def _polyline(points, t_total):
    """Равномерное движение по ломаной за t_total секунд."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    def path(t):
        d = min(max(t / t_total, 0.0), 1.0) * cum[-1]
        k = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
        a = (d - cum[k]) / max(seg[k], 1e-9)
        return tuple(pts[k] + a * (pts[k + 1] - pts[k]))
    return path


def test_route_from_main_road_to_lower_exit_is_illegal_turn():
    """С магистрали вглубь площади и разворот на нижний конец ближнего
    перехода — нарушение (C3896, 45: 49.6-54.6 с); тот же выезд машиной,
    заехавшей на площадь справа, — нет (C3905, 457)."""
    q = point_in("queue_zone", seed=31)
    far = point_in("crossing_far", seed=32)
    apex = point_in("illegal_turn_apex", exclude=PED_SAFE + ("illegal_turn_exit",), seed=33)
    out = point_in("illegal_turn_exit", seed=34)
    route = _polyline([q, far, apex, out], 12.0)
    assert len(events_of(track(1, 2, route, 0, 12, w=300, h=200), "illegal_turn")) == 1
    right_side = _polyline([apex, out], 6.0)
    assert events_of(track(2, 2, right_side, 0, 6, w=300, h=200), "illegal_turn") == []
    # с магистрали, но без разворота на площади: вдоль нижнего края кадра справа
    # налево (C3897, 602 — трекер склеил две машины)
    bottom = _polyline([q, far, (3700, 2150), (1500, 2150)], 12.0)
    assert events_of(track(3, 2, bottom, 0, 12, w=300, h=200), "illegal_turn") == []


def test_rider_on_motorcycle_is_not_jaywalking():
    p = point_in("crossroad", exclude=PED_SAFE, seed=3)
    path = lambda t: (p[0] + 60 * t, p[1])
    person = track(1, 0, path, 0, 5, w=40, h=90)
    bike = track(2, 3, path, 0, 5, w=90, h=110)
    assert events_of(person + bike, "jaywalking") == []


def _red_then_green(t_green, t_end):
    return [[round(t, 2), "red" if t < t_green else "green"] for t in np.arange(0, t_end, 0.12)]


def test_stop_on_the_zebra_on_red_is_stop_line():
    """Встал передом на зебре на красный и поехал на зелёный (C3905, 1:18):
    stop_line, не red_light и не stopped_vehicle."""
    p = point_in("past_stop_line", exclude=("stop_line",), seed=31)      # на зебре, в полосах очереди
    q = point_in("crossroad", exclude=("crossing_far", "past_stop_line"), seed=32)
    go = 30.0
    path = lambda t: p if t < go else (p[0] + (q[0] - p[0]) * min(1, (t - go) / 3),
                                       p[1] + (q[1] - p[1]) * min(1, (t - go) / 3))
    recs = track(1, 2, path, 0, 36, w=300, h=200)
    stitch_records(recs)
    ev = rules.compute_events(recs, ZONES, _red_then_green(go - 0.5, 36))
    labels = [e[2] for e in ev]
    assert labels.count("stop_line") == 1 and "red_light" not in labels, ev
    s = [e for e in ev if e[2] == "stop_line"][0]
    assert s[0] < 1.0 and abs(s[1] - (go - 0.5)) < 0.3          # до включения зелёного


def test_waiting_past_the_line_then_going_on_green_is_not_red_light():
    """Курьер ждёт за стоп-линией и уезжает с очередью на зелёный (C3902, 1:38)."""
    a = point_in("stop_line", seed=33)
    b = point_in("past_stop_line", exclude=("stop_line",), seed=34)
    q = point_in("crossroad", exclude=("crossing_far", "past_stop_line"), seed=35)
    def path(t):
        if t < 10:
            return a
        if t < 12:
            k = (t - 10) / 2
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
        if t < 20:
            return b
        k = min(1, (t - 20) / 3)
        return (b[0] + (q[0] - b[0]) * k, b[1] + (q[1] - b[1]) * k)
    recs = track(1, 3, path, 0, 25, w=60, h=80)
    stitch_records(recs)
    labels = [e[2] for e in rules.compute_events(recs, ZONES, _red_then_green(19.5, 25))]
    assert "red_light" not in labels and "stop_line" in labels, labels


def test_driving_through_on_red_is_red_light():
    """Проехал стоп-линию и зебру на красный без остановки: red_light, не stop_line."""
    a = point_in("queue_zone", seed=36)
    q = point_in("crossroad", exclude=("crossing_far", "past_stop_line"), seed=37)
    path = lambda t: (a[0] + (q[0] - a[0]) * min(1, t / 4), a[1] + (q[1] - a[1]) * min(1, t / 4))
    recs = track(1, 2, path, 0, 8, w=300, h=200)
    stitch_records(recs)
    labels = [e[2] for e in rules.compute_events(recs, ZONES, _red_then_green(99, 8))]
    assert labels.count("red_light") == 1 and "stop_line" not in labels, labels
