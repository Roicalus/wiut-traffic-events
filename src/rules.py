"""
rules.py — правила событий поверх сшитых треков (track.py -> stitch_tracks.py),
зон сцены (zones.json, совмещённых с видео в align.py) и состояния светофора
(light_state.py). Вызывается из pipeline.infer(); точка входа —
compute_events() (сабмит) / compute_events_debug() (то же с id объектов).
Отправляемые классы — solution.CLASSES, остальные правила экспериментальные.

CLI main() — только для отладки на сохранённых JSON треков:
    python src/rules.py --tracks src/tracks/C3896.stitched.json --zones zones.json \
        --light src/light/C3896.json --out predictions/C3896.json

Пороги подобраны по сэмплам (см. docs/CHANGES.md). Скорость нормируется на диагональ
бокса объекта (грубая компенсация перспективы: у дальних машин те же
пиксели/сек означают куда большую реальную скорость, чем у ближних), так
что пороги — это "длины корпуса в секунду", не px/s. Калибруй по своей
разметке:
    python evaluate.py --pred predictions.json --gt ground_truth.json --per-video
"""
import argparse
import bisect
import json
from pathlib import Path

import cv2
import numpy as np

VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck
PERSON_CLASS = 0


# ---------------------------------------------------------------- helpers
def load_tracks(path):
    data = json.loads(Path(path).read_text())
    return data["meta"], data["tracks"]


def load_zones(path):
    raw = json.loads(Path(path).read_text())
    return {name: np.array(pts, dtype=np.float32) for name, pts in raw.items()}


def in_zone(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0


def group_by_object(records):
    groups = {}
    for r in records:
        oid = r.get("stitched_id", r["track_id"])
        groups.setdefault(oid, []).append(r)
    for recs in groups.values():
        recs.sort(key=lambda r: r["t_sec"])
    return groups


# Точка, по которой объект "стоит" в зоне. Зоны размечены по асфальту, а
# центр бокса у машины/человека висит над землёй и в перспективе камеры
# сдвинут "дальше" от реального места — у пешехода на тротуаре центр
# торса может уже попадать в roadway. Низ-центр бокса (колёса/ноги) —
# правильная проекция на плоскость дороги. "center" — старое поведение.
ZONE_ANCHOR = "bottom"
# Скорость считаем по смещению за окно, а не между соседними сэмплами:
# при stride=3 (0.12 c) дрожание бокса на 2-3 px у дальней машины уже даёт
# ~0.3 "корпуса/с" — как раз порог остановки, и стоящая машина "едет",
# рвя stopped_vehicle/congestion на куски.
SPEED_WINDOW_SEC = 0.6


def _anchor(r):
    cx = (r["x1"] + r["x2"]) / 2.0
    if ZONE_ANCHOR == "bottom":
        return (cx, r["y2"])
    return (cx, (r["y1"] + r["y2"]) / 2.0)


def annotate(recs, zones, speed_window=SPEED_WINDOW_SEC):
    """Траектория объекта: на каждый сэмпл
      t      — время;
      c      — центр бокса (для попарных расстояний между объектами);
      g      — опорная точка на земле (ZONE_ANCHOR) — по ней зоны и линии;
      speed, vx, vy — скорость в "длинах корпуса в секунду" (нормировка
               на диагональ бокса = грубая компенсация перспективы),
               по смещению центра за окно speed_window;
      diag   — диагональ бокса в px (для попарной нормировки);
      zones  — множество зон, в которых лежит g;
      cls    — класс COCO.
    """
    samples = []
    ts, cs = [], []
    j = 0
    for r in recs:
        c = ((r["x1"] + r["x2"]) / 2.0, (r["y1"] + r["y2"]) / 2.0)
        g = _anchor(r)
        diag = max(((r["x2"] - r["x1"]) ** 2 + (r["y2"] - r["y1"]) ** 2) ** 0.5, 1.0)
        t = r["t_sec"]
        ts.append(t)
        cs.append(c)
        # самый поздний сэмпл, который старше t хотя бы на speed_window
        while j + 1 < len(ts) - 1 and t - ts[j + 1] >= speed_window:
            j += 1
        speed = vx = vy = 0.0
        if len(ts) > 1:
            k = j if t - ts[j] >= speed_window else 0
            dt = t - ts[k]
            if dt > 0:
                vx = (c[0] - cs[k][0]) / diag / dt
                vy = (c[1] - cs[k][1]) / diag / dt
                speed = (vx ** 2 + vy ** 2) ** 0.5
        zones_here = {name for name, poly in zones.items()
                      if len(poly) >= 3 and in_zone(g, poly)}
        samples.append({"t": t, "c": c, "g": g, "speed": speed, "vx": vx, "vy": vy,
                        "diag": diag, "zones": zones_here, "cls": r["cls"]})
    return samples


def sample_runs(samples, predicate, max_gap):
    """Интервалы (start, end), где predicate(sample) истинно на
    последовательных сэмплах; разрывает бег, если между сэмплами дыра >
    max_gap (объект пропал из трекинга — не факт, что состояние
    сохранилось всё это время)."""
    runs, start, prev_t = [], None, None
    for s in samples:
        ok = predicate(s)
        if prev_t is not None and s["t"] - prev_t > max_gap:
            if start is not None:
                runs.append((start, prev_t))
            start = s["t"] if ok else None
        elif ok and start is None:
            start = s["t"]
        elif not ok and start is not None:
            runs.append((start, prev_t))
            start = None
        prev_t = s["t"]
    if start is not None:
        runs.append((start, prev_t))
    return runs


def merge_intervals(intervals, gap=0.0):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def concurrent_runs(intervals, thresh):
    """Интервалы времени, где одновременно активно >= thresh интервалов
    (sweep line по +1/-1 событиям начала/конца)."""
    if not intervals:
        return []
    # при равном t сначала +1: передача "эстафеты" (одна машина уехала, другая
    # встала в тот же сэмпл) и интервал нулевой длины не роняют счётчик
    events = sorted([(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals],
                    key=lambda x: (x[0], -x[1]))
    count, start, runs = 0, None, []
    for t, delta in events:
        count += delta
        if count >= thresh and start is None:
            start = t
        elif count < thresh and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, events[-1][0]))
    return runs


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def first_entry_time(samples, zone_name):
    was_in = False
    for s in samples:
        now_in = zone_name in s["zones"]
        if now_in and not was_in:
            return s["t"]
        was_in = now_in
    return None


def _angle_between(vx1, vy1, vx2, vy2):
    """Угол между двумя векторами в градусах [0, 180]. None, если один из
    векторов нулевой (объект стоит — направление не определено)."""
    n1 = (vx1 ** 2 + vy1 ** 2) ** 0.5
    n2 = (vx2 ** 2 + vy2 ** 2) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = max(-1.0, min(1.0, (vx1 * vx2 + vy1 * vy2) / (n1 * n2)))
    return np.degrees(np.arccos(cos_a))


def _line_side(point, line):
    """Знак векторного произведения — по какую сторону от line (2 точки)
    лежит point. line — [[x1,y1],[x2,y2]] (как размечает define_zones.py
    для solid_line*: ЛИНИЯ, не полигон, поэтому pointPolygonTest не
    применим — у линии нет "внутри", есть только две стороны)."""
    (x1, y1), (x2, y2) = line[0], line[1]
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _closest_approach(samples_a, samples_b, max_dt=0.6):
    """Совмещает две траектории по времени (разные объекты сэмплируются
    по одной и той же сетке кадров видео, но могут пропадать из трекинга
    в разные моменты — поэтому ищем ближайший по времени сэмпл, а не
    требуем точного совпадения индекса).

    Возвращает список (t, dist_norm, closing) отсортированный по t:
      dist_norm — расстояние между центроидами / средняя диагональ пары
                  ("длины корпуса", та же нормировка, что у speed);
      closing   — скорость сближения (dist_norm убывает -> closing > 0),
                  None для самой первой точки пары (нет предыдущей)."""
    ts_b = [s["t"] for s in samples_b]
    aligned = []
    for sa in samples_a:
        i = bisect.bisect_left(ts_b, sa["t"])
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(samples_b):
                sb = samples_b[j]
                dt = abs(sb["t"] - sa["t"])
                if dt <= max_dt and (best is None or dt < best[0]):
                    best = (dt, sb)
        if best is None:
            continue
        _, sb = best
        t = (sa["t"] + sb["t"]) / 2.0
        avg_diag = max((sa["diag"] + sb["diag"]) / 2.0, 1.0)
        dist = ((sa["c"][0] - sb["c"][0]) ** 2 + (sa["c"][1] - sb["c"][1]) ** 2) ** 0.5
        aligned.append((t, dist / avg_diag))

    aligned.sort(key=lambda x: x[0])
    out = []
    prev = None
    for t, dist_norm in aligned:
        closing = None
        if prev is not None:
            pt, pd = prev
            dt = t - pt
            if dt > 1e-6:
                closing = (pd - dist_norm) / dt
        out.append((t, dist_norm, closing))
        prev = (t, dist_norm)
    return out


class LightState:
    def __init__(self, samples):
        """samples: список [t_sec, state], как возвращает
        light_state.run_light_state() (или читается из его JSON)."""
        samples = samples or []
        self.ts = [t for t, _ in samples]
        self.states = [s for _, s in samples]

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text()) if path else {"samples": []}
        return cls(data["samples"])

    def at(self, t):
        if not self.ts:
            return "unknown"
        i = bisect.bisect_right(self.ts, t) - 1
        i = max(0, min(i, len(self.states) - 1))
        return self.states[i]

    def next_after(self, t, state):
        for tt, s in zip(self.ts, self.states):
            if tt > t and s == state:
                return tt
        return None


# ---------------------------------------------------------------- rules
GREEN_STAND_SEC = 15.0     # очередь стоит на зелёном дольше — она не рассосалась за фазу
JUNCTION_JAM_SEC = 40.0    # плотная масса на перекрёстке дольше фазы светофора (~37 с)


def _stationary_clusters(vehicle_objs, zones, min_count, merge_gap, min_duration,
                         speed_thresh, max_gap):
    """Интервалы, когда в зонах zones одновременно стоят >= min_count машин."""
    intervals = []
    for samples in vehicle_objs.values():
        intervals += sample_runs(
            samples, lambda x: x["speed"] < speed_thresh and bool(zones & x["zones"]), max_gap)
    active = merge_intervals(concurrent_runs(intervals, min_count), gap=merge_gap)
    return [iv for iv in active if iv[1] - iv[0] >= min_duration]   # после склейки


def _green_time(light, s, e):
    """Сколько секунд из [s, e] горел зелёный (по сэмплам светофора)."""
    if light is None or not light.ts:
        return 0.0
    lo, hi = bisect.bisect_left(light.ts, s), bisect.bisect_right(light.ts, e)
    ts, st = light.ts[lo:hi], light.states[lo:hi]
    return sum(t1 - t0 for t0, t1, x in zip(ts, ts[1:], st) if x == "green")


def detect_congestion(vehicle_objs, signal_zones=("queue_zone",), junction_zones=(), light=None,
                      min_count=4, min_duration=5.0, merge_gap=3.0, speed_thresh=0.3, max_gap=2.0,
                      green_stand_sec=GREEN_STAND_SEC, junction_jam_sec=JUNCTION_JAM_SEC):
    """Затор (определение задания: поток стоит или ползёт по всем полосам
    направления; от "очередь встала" до "очередь рассосалась").

    Очередь на сигнал — не затор: она разъезжается на каждом зелёном. На
    сэмплах кластеры из 4+ стоящих машин в queue_zone приходятся на красный
    (зелёный — 10-37% их времени: это задержка разъезда), и прежнее правило
    "4+ машины стоят" превращало в congestion каждую фазу красного.
      * очередь перед светофором (signal_zones): затор, только если 4+
        машины простояли >= green_stand_sec ПРИ ЗЕЛЁНОМ. Без светофора —
        как на перекрёстке, по длительности;
      * сам перекрёсток за светофором (junction_zones, crossroad*): затор,
        если плотная масса стоит дольше фазы (>= junction_jam_sec) —
        C3896, 40-108 с.
    Кластеры считаются по группам зон отдельно: очередь и площадь — разные
    места, и сумма "2 стоят там + 2 тут" — не затор."""
    signal_zones, junction_zones = set(signal_zones), set(junction_zones)
    args = (min_count, merge_gap, min_duration, speed_thresh, max_gap)
    jams = []
    for s, e in _stationary_clusters(vehicle_objs, signal_zones, *args):
        if (light is not None and light.ts and _green_time(light, s, e) >= green_stand_sec) or \
                ((light is None or not light.ts) and e - s >= junction_jam_sec):
            jams.append((s, e))
    if junction_zones:
        jams += [(s, e) for s, e in _stationary_clusters(vehicle_objs, junction_zones, *args)
                 if e - s >= junction_jam_sec]
    return [[round(s, 2), round(e, 2), "congestion"] for s, e in merge_intervals(sorted(jams))]


SIGNAL_WAIT_ZONES = ("queue_zone", "stop_line", "crossing_far")
GREEN_RELEASE_SEC = 8.0


def _released_by_green(light, t_stop, t_move, window=GREEN_RELEASE_SEC):
    """Машина тронулась в течение window c после переключения на зелёный,
    а стояла в основном на красном -> это ожидание сигнала."""
    if light is None or not light.ts:
        return False
    t_green = light.next_after(t_stop, "green")
    if t_green is None or not (t_green - 1.0 <= t_move <= t_green + window):
        return False
    return light.at(t_stop + 0.5) in ("red", "yellow")


def _stationary_runs(vehicle_objs, speed_thresh, max_gap):
    """oid -> [(start, end, центр px, диагональ px)] — стоянки объекта."""
    out = {}
    for oid, samples in vehicle_objs.items():
        runs = []
        for s, e in sample_runs(samples, lambda x: x["speed"] < speed_thresh, max_gap):
            inside = [x for x in samples if s - 1e-6 <= x["t"] <= e + 1e-6]
            c = np.median([x["c"] for x in inside], axis=0)
            runs.append((s, e, c, float(np.median([x["diag"] for x in inside]))))
        out[oid] = runs
    return out


def _time_with_at_least(intervals, k):
    """Суммарное время, когда одновременно открыто >= k интервалов."""
    edges = sorted([(a, 1) for a, b in intervals if b > a] + [(b, -1) for a, b in intervals if b > a])
    total, depth, prev = 0.0, 0, None
    for t, d in edges:
        if depth >= k and prev is not None:
            total += t - prev
        depth += d
        prev = t
    return total


def detect_stopped_vehicle(vehicle_objs, road_zones=None, junction_zones=(),
                            speed_thresh=0.3, min_duration=10.0, queue_min_duration=75.0,
                            max_gap=2.0, cluster_min_others=2, cluster_radius=1.5,
                            cluster_overlap_frac=0.5, on_road_frac=0.5, light=None,
                            return_ids=False):
    """Стоит >=10с (или >=queue_min_duration, если стоит в queue_zone — это
    обычное ожидание цикла светофора, не событие, пока не затянулось) и это
    НЕ часть пробки или очереди.

    Часть очереди, пробки или ряда припаркованных — если >= cluster_overlap_frac
    времени стоянки ВПЛОТНУЮ к ней одновременно стоят >= cluster_min_others
    машин (в заторе соседи приходят и уходят — считаем время, а не
    соседей, простоявших всю стоянку целиком). Вплотную — не дальше
    cluster_radius диагоналей МЕНЬШЕГО из двух боксов (у машины вблизи камеры
    диагональ огромная, и по большей "соседями" становилась очередь на
    магистрали в 650 px): очередь — это цепочка машин вплотную. Раньше машину отбрасывало
    пересечение с ЛЮБЫМ сегментом congestion где угодно в кадре, и одиночная
    машина, простоявшая 36 с посреди площади (C3897, 210-246 с), пропадала,
    потому что в это время на магистрали стояла очередь на красный.
    Проверка "в той же зоне" не лучше: зона площади — полкадра, и там почти
    всегда кто-то ждёт выезда.

    junction_zones — сам перекрёсток ЗА светофором (crossroad*). Для машины,
    простоявшей там, не действует исключение "ждала зелёного": стоять
    посреди перекрёстка нельзя, даже если трогаешься вместе с фазой (C3897:
    0-27 с и 210-250 с — одиночные машины тронулись через 1-5 с после
    зелёного). Исключение "стоит в кластере" остаётся: плотная масса машин
    на площади — это congestion (C3896, 40-70 с), а не десяток stopped_vehicle.

    Раньше здесь не было отдельного порога для queue_zone: машина,
    отстоявшая один нормальный цикл красного (15-40с) в очереди, но не
    набравшая критическую массу для congestion (см. detect_congestion,
    min_count), ложно засчитывалась как stopped_vehicle. Порог
    queue_min_duration (60-90с по заданию) фильтрует это.

    road_zones: множество имён зон, которые физически являются проезжей
    частью (roadway*, crossroad*, queue_zone, stop_line, crossing_*, см.
    _scene_zone_sets). Если задано и меньше on_road_frac (по умолчанию
    половины) сэмплов бега остановки попадает в эти зоны — событие НЕ
    пишется: скорее всего это машина, стоящая на обочине/парковке/за
    пределами размеченной дороги, а не "встала посреди проезжей части"."""
    junction_zones = set(junction_zones)
    stationary = _stationary_runs(vehicle_objs, speed_thresh, max_gap)
    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        for s, e, c, diag in stationary[oid]:
            dur = e - s
            run_samples = [x for x in samples if s - 1e-6 <= x["t"] <= e + 1e-6]
            if road_zones:
                frac_on_road = (sum(1 for x in run_samples if road_zones & x["zones"])
                                  / max(len(run_samples), 1))
                if frac_on_road < on_road_frac:
                    continue
            # Первая машина очереди стоит не в queue_zone, а на стоп-линии или
            # прямо на переходе — это тоже ожидание сигнала, не stopped_vehicle.
            frac_in_queue = (sum(1 for x in run_samples if x["zones"] & set(SIGNAL_WAIT_ZONES))
                              / max(len(run_samples), 1))
            required = queue_min_duration if frac_in_queue > 0.5 else min_duration
            if dur < required:
                continue
            frac_junction = (sum(1 for x in run_samples if x["zones"] & junction_zones)
                             / max(len(run_samples), 1))
            on_junction = frac_junction > 0.5 and frac_in_queue <= 0.5
            if not on_junction and _released_by_green(light, s, e):
                continue
            near = [(max(s, os_), min(e, oe)) for other, runs in stationary.items() if other != oid
                    for os_, oe, oc, od in runs
                    if os_ < e and oe > s and np.hypot(*(oc - c)) <= cluster_radius * min(diag, od)]
            if _time_with_at_least(near, cluster_min_others) >= cluster_overlap_frac * dur:
                continue
            events.append([round(s, 2), round(e, 2), "stopped_vehicle"])
            ids.append(oid)
    return (events, ids) if return_ids else events


def _zones_by_prefix(zones, prefixes):
    """Имена зон, совпадающие с одним из prefixes целиком или начинающиеся
    с 'prefix_'. Так один физический смысл ("проезжая часть", "легитимный
    переход") можно разбить на несколько полигонов под неудобную форму
    сцены (roadway, roadway_before_queue, crossroad, crossroad_2 — все
    "проезжая часть"; crossing_far, crossing_near — все "переход"),
    вместо одного самопересекающегося полигона, который cv2 не умеет."""
    names = set()
    for name in zones:
        for p in prefixes:
            if name == p or name.startswith(p + "_"):
                names.add(name)
                break
    return names


RIDER_IOA = 0.6   # доля бокса человека внутри бокса транспорта: едет на нём/в нём


def _mark_riders(groups, person_objs):
    """sample["in_vehicle"] у людей, чей бокс в этом кадре почти целиком внутри
    бокса транспорта: мотоциклист, велосипедист, пассажир у окна автобуса
    (C3902, 145 с: мотоциклист был jaywalking)."""
    boxes = lambda rs: np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rs], np.float32)
    vehicles, persons = {}, {}
    for oid, recs in groups.items():
        target = vehicles if recs[0]["cls"] in VEHICLE_CLASSES else persons if oid in person_objs else None
        if target is not None:
            for k, r in enumerate(recs):
                target.setdefault(r["frame"], []).append((r, oid, k))
    for smp_list in person_objs.values():
        for smp in smp_list:
            smp["in_vehicle"] = False
    for frame, plist in persons.items():
        vlist = vehicles.get(frame)
        if not vlist:
            continue
        P, V = boxes([p[0] for p in plist]), boxes([v[0] for v in vlist])
        iw = np.clip(np.minimum(P[:, None, 2], V[None, :, 2]) - np.maximum(P[:, None, 0], V[None, :, 0]), 0, None)
        ih = np.clip(np.minimum(P[:, None, 3], V[None, :, 3]) - np.maximum(P[:, None, 1], V[None, :, 1]), 0, None)
        area = np.maximum((P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1]), 1.0)
        rider = ((iw * ih) >= RIDER_IOA * area[:, None]).any(axis=1)
        for (_r, oid, k), is_rider in zip(plist, rider):
            person_objs[oid][k]["in_vehicle"] = bool(is_rider)


def detect_jaywalking(person_objs, roadway_zones, crossing_zones=(), min_duration=1.0,
                       max_gap=1.0, return_ids=False):
    """Пешеход на проезжей части ВНЕ легитимного перехода.

    roadway_zones/crossing_zones — множества имён зон (см. _zones_by_prefix).
    crossing_zones обязательно исключаются: на реальной разметке зоны
    "проезжая часть" и "переход" почти всегда немного перекрываются на
    границе (тут так и есть — crossroad_2 залезает на crossing_near), и
    без явного исключения человек, идущий ПО зебре в зоне нахлёста, тоже
    засчитывался бы как jaywalking."""
    crossing_zones = set(crossing_zones)
    events, ids = [], []
    for oid, samples in person_objs.items():
        runs = sample_runs(
            samples,
            lambda s: (bool(roadway_zones & s["zones"]) and not (crossing_zones & s["zones"])
                       and not s.get("in_vehicle")),
            max_gap,
        )
        for s, e in runs:
            if e - s >= min_duration:
                events.append([round(s, 2), round(e, 2), "jaywalking"])
                ids.append(oid)
    return (events, ids) if return_ids else events


def detect_red_light(vehicle_objs, light, gate_zone="stop_line", exit_zone="crossing_far",
                      return_ids=False):
    """Машина реально проезжает на красный: заходит в gate_zone (широкая
    зона очереди перед переходом, это ок и на "правильную" остановку) на
    красный свет И ДОЕЗЖАЕТ до exit_zone, пока свет ВСЁ ЕЩЁ красный.

    Раньше свет проверялся только один раз — в момент входа в gate_zone —
    и событие писалось, даже если машина: (а) вообще не доехала до
    exit_zone за время трека (просто стояла в очереди), или (б) доехала
    до exit_zone уже на зелёном, честно отстояв цикл светофора. Оба
    случая давали массовые false positive почти на каждой машине в
    очереди, т.к. gate_zone у нас широкая (весь фронт очереди), а не
    тонкая линия."""
    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        t_cross = first_entry_time(samples, gate_zone)
        if t_cross is None or light.at(t_cross) != "red":
            continue
        t_end, was_in_exit, violated = None, False, False
        for s in samples:
            if s["t"] < t_cross:
                continue
            if exit_zone in s["zones"]:
                if not was_in_exit:
                    # момент фактического въезда в exit_zone — свет должен
                    # быть красным именно сейчас, а не только при t_cross
                    violated = light.at(s["t"]) == "red"
                was_in_exit = True
            elif was_in_exit:
                t_end = s["t"]
                break
        if not was_in_exit or not violated:
            continue  # не доехал(а) до exit_zone, или доехал(а) уже на зелёном
        if t_end is None:
            t_end = samples[-1]["t"]
        if t_end > t_cross:
            events.append([round(t_cross, 2), round(t_end, 2), "red_light"])
            ids.append(oid)
    return (events, ids) if return_ids else events


def detect_stop_line(vehicle_objs, light, speed_thresh=0.3, max_gap=2.0, return_ids=False):
    """Бонус: та же light-детекция почти бесплатно даёт stop_line —
    остановка ПЕРЕД stop_line на красный (машина не заезжает на переход)."""
    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        runs = sample_runs(samples, lambda s: s["speed"] < speed_thresh and "stop_line" in s["zones"], max_gap)
        for s, e in runs:
            if light.at(s) != "red":
                continue
            t_green = light.next_after(s, "green")
            t_end = t_green if t_green is not None else e
            events.append([round(s, 2), round(t_end, 2), "stop_line"])
            ids.append(oid)
    return (events, ids) if return_ids else events


# Зоны с ОДНИМ направлением движения, где wrong_way вообще имеет смысл.
# Площадь/перекрёсток сюда не входят: там законно едут в разные стороны.
WRONG_WAY_ZONES = ("roadway", "roadway_before_queue")
WRONG_WAY_MIN_SAMPLES = 200
WRONG_WAY_MIN_CONCENTRATION = 0.6


def detect_wrong_way(vehicle_objs, roadway_zones, min_speed=0.15, angle_thresh=140.0,
                      min_duration=1.0, max_gap=1.5, return_ids=False):
    """Машина едет против преобладающего потока на проезжей части.

    ПЕРВАЯ версия, не откалибрована. Вместо жёстко зашитого вектора
    направления (который пришлось бы подбирать вручную под ракурс этой
    конкретной камеры) эталонное направление потока считается САМ ИЗ
    ВИДЕО: циркулярное среднее векторов скорости всех машин на
    проезжей части в этом же ролике. Это устойчивее к неточной ручной
    прикидке направления по camera_own.md и подходит для скрытого теста
    той же камеры без переразметки. Компромисс: если в ролике реально
    почти все едут "неправильно" (сам поток развернули, например
    ремонт/перекрытие), эталон сместится и wrong_way не сработает —
    в сэмплах организаторов такого не замечено (camera_own.md).

    angle_thresh=140° — велик специально: обычные манёвры (перестроение,
    поворот на площади) меняют курс на 30-90°, встречное движение — это
    ~180°. Порог ближе к 180, чем к 90, чтобы не путать поворот с wrong_way."""
    # Эталон считается ОТДЕЛЬНО для каждой зоны и только там, где поток
    # однонаправленный. На площади (crossroad*) сходятся несколько дорог:
    # единый усреднённый "эталон" объявлял встречным обычный поток с
    # правой дороги (видно на debug-видео C3896/C3902).
    zones_used = [z for z in WRONG_WAY_ZONES if z in roadway_zones]
    if not zones_used:
        return ([], []) if return_ids else []

    refs = {}
    for zname in zones_used:
        sx = sy = 0.0
        n = 0
        for samples in vehicle_objs.values():
            for s in samples:
                if s["speed"] >= min_speed and zname in s["zones"]:
                    sx += s["vx"] / max(s["speed"], 1e-6)
                    sy += s["vy"] / max(s["speed"], 1e-6)
                    n += 1
        if n < WRONG_WAY_MIN_SAMPLES:
            continue
        concentration = (sx ** 2 + sy ** 2) ** 0.5 / n   # 1 = все в одну сторону
        if concentration >= WRONG_WAY_MIN_CONCENTRATION:
            refs[zname] = (sx, sy)
        else:
            print(f"wrong_way: зона {zname} пропущена — поток не однонаправленный "
                  f"(концентрация {concentration:.2f})")
    if not refs:
        return ([], []) if return_ids else []

    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        def against_flow(s):
            if s["speed"] < min_speed:
                return False
            for zname, (rx, ry) in refs.items():
                if zname in s["zones"]:
                    ang = _angle_between(s["vx"], s["vy"], rx, ry)
                    return ang is not None and ang >= angle_thresh
            return False

        for s, e in sample_runs(samples, against_flow, max_gap):
            if e - s >= min_duration:
                events.append([round(s, 2), round(e, 2), "wrong_way"])
                ids.append(oid)
    return (events, ids) if return_ids else events


def _heading_series(samples, min_speed):
    """[(t, heading_deg)] только для сэмплов, где объект реально движется
    (иначе heading — шум формата atan2(0,0))."""
    out = []
    for s in samples:
        if s["speed"] >= min_speed:
            out.append((s["t"], float(np.degrees(np.arctan2(s["vy"], s["vx"])))))
    return out


def detect_illegal_u_turn(vehicle_objs, roadway_zones, min_speed=0.12, window_sec=4.0,
                           turn_angle_thresh=120.0, max_gap=2.0, return_ids=False):
    """Разворот на 180° на проезжей части. ПЕРВАЯ версия: разметки "здесь
    разворот разрешён" нет нигде в zones.json/camera_own.md, поэтому —
    — ЛЮБОЙ обнаруженный
    разворот на проезжей части (roadway*/crossroad*) считается illegal.
    Если на скрытом тесте окажется место с разрешённым разворотом, это
    даст FP — придётся завести отдельную зону-исключение по образцу
    illegal_turn_exit.

    Детектируем не сам "разворот" геометрически (нет разметки полос по
    сторонам), а ФАКТ разворота курса: heading в момент t2 отличается от
    heading в t1 на >= turn_angle_thresh, окно t2-t1 <= window_sec (разворот
    — манёвр за несколько секунд, не мгновенный), и весь промежуток
    объект остаётся на проезжей части (иначе это могло бы быть, например,
    заездом за пределы кадра и появлением с другим курсом — другой физический
    смысл)."""
    if not roadway_zones:
        return ([], []) if return_ids else []

    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        heading = _heading_series(samples, min_speed)
        # индекс сэмпла (по времени) -> находится ли объект на проезжей части
        on_road_at = {s["t"]: bool(roadway_zones & s["zones"]) for s in samples}
        ts_all = sorted(on_road_at)

        candidates = []
        for i in range(len(heading)):
            t1, h1 = heading[i]
            for j in range(i + 1, len(heading)):
                t2, h2 = heading[j]
                dt = t2 - t1
                if dt > window_sec:
                    break
                if dt < window_sec * 0.35:
                    continue  # слишком быстро для реального разворота — скорее шум курса
                diff = abs(h1 - h2)
                diff = min(diff, 360.0 - diff)
                if diff >= turn_angle_thresh:
                    # весь диапазон [t1, t2] должен быть на проезжей части
                    span = [t for t in ts_all if t1 - 1e-6 <= t <= t2 + 1e-6]
                    if span and all(on_road_at[t] for t in span):
                        candidates.append((t1, t2))
        merged = merge_intervals(candidates, gap=max_gap)
        for s, e in merged:
            events.append([round(s, 2), round(e, 2), "illegal_u_turn"])
            ids.append(oid)
    return (events, ids) if return_ids else events


TWO_WHEELERS = {1, 3}         # bicycle, motorcycle
# Заезд на тротуарный островок. Официального класса для него нет (добавлять
# свои id нельзя — харнесс их выбросит), поэтому это ДИАГНОСТИЧЕСКОЕ событие:
# видно в визуализации, в predictions.json не попадает (solution.DIAGNOSTIC_CLASSES).
CURB_MOUNT = "curb_mount"
ISLAND_FOOTPRINT_FRAC = 0.3   # доля высоты бокса над его низом: середина колёсной базы
TURN_RATE_DEG_S = 12.0        # курс меняется быстрее — машина в повороте
TURN_PAD_MIN, TURN_PAD_MAX = 1.0, 4.0


def _turn_span(samples, t0, t1, min_speed=0.3):
    """Границы поворота вокруг [t0, t1]: пока курс меняется быстрее
    TURN_RATE_DEG_S, но не меньше TURN_PAD_MIN и не больше TURN_PAD_MAX с
    каждой стороны. В разметке поворот — от начала до конца манёвра, а
    наезд на островок — только его середина (0.4 с): без расширения IoU < 0.3."""
    hs = _heading_series(samples, min_speed)
    if len(hs) < 3:
        return t0 - TURN_PAD_MIN, t1 + TURN_PAD_MIN
    ts = np.array([t for t, _ in hs])
    hd = np.unwrap(np.radians([h for _, h in hs]))
    rate = np.degrees(np.abs(np.gradient(hd, ts)))
    start, end = t0 - TURN_PAD_MIN, t1 + TURN_PAD_MIN
    for k in range(int(np.searchsorted(ts, t0)) - 1, -1, -1):
        if rate[k] < TURN_RATE_DEG_S or t0 - ts[k] > TURN_PAD_MAX:
            break
        start = min(start, ts[k])
    for k in range(int(np.searchsorted(ts, t1)), len(ts)):
        if rate[k] < TURN_RATE_DEG_S or ts[k] - t1 > TURN_PAD_MAX:
            break
        end = max(end, ts[k])
    return float(max(start, samples[0]["t"])), float(min(end, samples[-1]["t"]))


ROUTE_TURN_DEV_DEG = 20.0   # курс отклонился от курса подъезда -> поворот начался
ROUTE_TURN_MIN_SPEED = 0.07  # медленнее — курс шумит (стоянка перед поворотом)
ROUTE_TURN_MAX_SEC = 6.0     # раньше — это перестроения в пробке на площади, не поворот


def _route_turn_span(samples, t_origin, t_junction, t_exit):
    """Поворот машины, приехавшей с магистрали: начало — первый сэмпл после
    въезда на площадь, где курс отклонился от курса подъезда больше чем на
    ROUTE_TURN_DEV_DEG; конец — въезд на выезд, продлённый, пока курс ещё
    меняется. Поворот часто начинается почти с места (C3896, 45: стояла
    40-48 с, повернула 49.6-55.9 с) — поэтому порог скорости низкий."""
    hs = _heading_series(samples, ROUTE_TURN_MIN_SPEED)
    approach = [h for t, h in hs if t_origin <= t <= t_junction]
    if not approach:
        return _turn_span(samples, t_exit, t_exit)
    ref = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(approach))))))
    dev = lambda h: abs((h - ref + 180.0) % 360.0 - 180.0)
    lo = max(t_junction, t_exit - ROUTE_TURN_MAX_SEC)
    start = next((t for t, h in hs if lo <= t <= t_exit and dev(h) > ROUTE_TURN_DEV_DEG),
                 t_exit - TURN_PAD_MIN)
    _, end = _turn_span(samples, t_exit, t_exit)
    return float(max(start, samples[0]["t"])), float(end)


def detect_curb_mount(vehicle_objs, zones, min_speed=0.3, min_duration=0.3, max_gap=1.0):
    """Машина заезжает на тротуарный островок (sidewalk*) на ходу.
    Низ бокса машины, едущей по диагонали, — это ближний к камере угол
    бампера, он остаётся на асфальте (C3905, 100 с: 0 из 7 кадров на
    островке); проверяем точку на ISLAND_FOOTPRINT_FRAC высоты бокса выше
    низа — там колёса. Возвращает ([события], [oid])."""
    islands = [np.asarray(zones[n], np.float32) for n in _zones_by_prefix(zones, ("sidewalk",))]
    events, ids = [], []
    if not islands:
        return events, ids

    def on_island(x):
        (cx, cy), (_, gy) = x["c"], x["g"]
        foot = (cx, gy - ISLAND_FOOTPRINT_FRAC * 2.0 * (gy - cy))   # h = 2 * (низ - центр)
        return any(in_zone(foot, poly) for poly in islands)

    for oid, samples in vehicle_objs.items():
        if samples[0]["cls"] in TWO_WHEELERS:
            continue  # мопед/велосипед заезжает на край островка законно (C3896, 24 с)
        runs = sample_runs(samples, lambda x: x["speed"] >= min_speed and on_island(x), max_gap)
        for s, e in runs:
            if e - s >= min_duration:
                s, e = _turn_span(samples, s, e)
                events.append([round(s, 2), round(e, 2), CURB_MOUNT])
                ids.append(oid)
    return events, ids


ILLEGAL_TURN_ORIGIN = ("queue_zone", "stop_line", "crossing_far")
APEX_TO_EXIT_MAX_SEC = 3.0


def detect_illegal_turn(vehicle_objs, zones, origin_zones=ILLEGAL_TURN_ORIGIN, return_ids=False):
    """Поворот в запрещённом направлении — по МАРШРУТУ, а не по месту поворота.

    На этом перекрёстке запрещённый манёвр — приехать с магистрали (очередь
    -> дальний переход), уйти вглубь площади, развернуться там (зона
    illegal_turn_apex*) и уехать назад-влево через нижний конец ближнего
    перехода (зона illegal_turn_exit*). Разрешённый поворот на ту же улицу —
    сразу направо, на верхний конец перехода. Проверено глазами на сэмплах:
    C3896 45 (49.6 с) и 830 (284 с) — нарушение. Без вершины разворота
    правило ловило машины, которые просто едут влево вдоль нижнего края
    кадра от правого края (C3896 628, C3897 602, C3902 x3: трекер склеил их
    с машиной с магистрали), и машины, заехавшие на площадь справа
    (C3905 457). Прежнее правило "поворот на 50-150 градусов внутри
    полигона" ловило объезды и пропускало эти развороты.

    Отрезок — сам поворот (_route_turn_span): от момента, когда курс
    отклонился от курса подъезда, до въезда в illegal_turn_exit (+ дотягиваем,
    пока курс ещё меняется). Заезд на островок — отдельно, detect_curb_mount."""
    events, ids = [], []
    exit_zones = _zones_by_prefix(zones, ("illegal_turn_exit",))
    apex_zones = _zones_by_prefix(zones, ("illegal_turn_apex",))
    junction = _zones_by_prefix(zones, ("crossroad",))
    origin = set(origin_zones)
    if not exit_zones or not apex_zones:
        return (events, ids) if return_ids else events
    for oid, samples in vehicle_objs.items():
        t_origin = next((x["t"] for x in samples if x["zones"] & origin), None)
        if t_origin is None:
            continue
        after = [x for x in samples if x["t"] > t_origin]
        t_apex = next((x["t"] for x in after if x["zones"] & apex_zones), None)
        if t_apex is None:
            continue
        t_exit = next((x["t"] for x in after if x["t"] > t_apex and x["zones"] & exit_zones), None)
        # разворот — это непрерывный манёвр: из вершины сразу в выезд (на сэмплах
        # 0.3-1 с). 6-8 с — склеенный трекером трек двух разных машин (C3902)
        if t_exit is not None and not any(
                x["zones"] & apex_zones for x in after if t_exit - APEX_TO_EXIT_MAX_SEC <= x["t"] < t_exit):
            continue
        if t_exit is None or not any(x["zones"] & junction for x in after if x["t"] < t_exit):
            continue
        t_junction = next(x["t"] for x in after if x["zones"] & junction)
        s, e = _route_turn_span(samples, t_origin, t_junction, t_exit)
        events.append([round(s, 2), round(e, 2), "illegal_turn"])
        ids.append(oid)
    return (events, ids) if return_ids else events


def detect_solid_line_crossing(vehicle_objs, zones, settle_sec=1.5, max_gap=2.0,
                                margin_norm=0.25, return_ids=False):
    """Пересечение сплошной линии — требует зон-ЛИНИЙ 'solid_line'/
    'solid_line_*' (ровно 2 точки каждая, см. define_zones.py, клавиша 7).
    Без такой линии в zones.json — ничего не находит, не падает.

    В отличие от полигональных зон, у линии нет "внутри" — cv2.pointPolygonTest
    не применим, поэтому сторона определяется знаком векторного произведения
    (_line_side). Пересечение — смена знака между последовательными сэмплами
    одного объекта, но с гистерезисом: сторона засчитывается, только если
    опорная точка отошла от линии дальше margin_norm диагоналей бокса —
    иначе машина, едущая ВДОЛЬ линии, от дрожания бокса "пересекает" её
    десятки раз. end = start + settle_sec (фиксированный запас на то,
    что объект "полностью в новой полосе" — секунда-полторы после пересечения
    в этой сцене соответствует масштабу машины на проезжей части; тонко
    настраивать без разметки machinery невозможно, это грубая, но безопасная
    оценка длительности события)."""
    lines = {name: pts for name, pts in zones.items()
             if name == "solid_line" or name.startswith("solid_line_")}
    if not lines:
        return ([], []) if return_ids else []

    events, ids = [], []
    for oid, samples in vehicle_objs.items():
        for line in lines.values():
            (x1, y1), (x2, y2) = line[0], line[1]
            seg_len = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)
            prev_side, prev_t = None, None
            for s_ in samples:
                if prev_t is not None and s_["t"] - prev_t > max_gap:
                    prev_side = None  # трек прерывался — не доверяем "пересечению" через дыру
                prev_t = s_["t"]
                # проекция на отрезок: за концами линии пересечения нет
                px, py = s_["g"]
                u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg_len ** 2
                if not 0.0 <= u <= 1.0:
                    continue
                dist = _line_side(s_["g"], line) / seg_len  # знаковое расстояние, px
                if abs(dist) < margin_norm * s_["diag"]:
                    continue  # в "мёртвой зоне" у линии — сторону не обновляем
                side = dist > 0
                if prev_side is not None and side != prev_side:
                    t_cross = s_["t"]
                    events.append([round(t_cross, 2), round(t_cross + settle_sec, 2),
                                   "solid_line_crossing"])
                    ids.append(oid)
                prev_side = side
    return (events, ids) if return_ids else events


def detect_failure_to_yield(vehicle_objs, person_objs, crossing_zones, max_gap=2.0,
                             ped_merge_gap=0.5, min_duration=0.2, return_ids=False):
    """Машина проезжает через переход, пока на нём (или заходит на него)
    пешеход. Считается ПО КАЖДОМУ переходу отдельно (crossing_far и
    crossing_near — разные физические переходы; не смешиваем пешехода на
    одном с машиной на другом)."""
    if not crossing_zones:
        return ([], []) if return_ids else []

    events, ids = [], []
    for zone in crossing_zones:
        ped_runs = []
        for samples in person_objs.values():
            ped_runs += sample_runs(samples, lambda s, z=zone: z in s["zones"], max_gap)
        ped_intervals = merge_intervals(ped_runs, gap=ped_merge_gap)
        if not ped_intervals:
            continue

        for oid, samples in vehicle_objs.items():
            veh_runs = sample_runs(samples, lambda s, z=zone: z in s["zones"], max_gap)
            for s, e in veh_runs:
                if e - s < min_duration:
                    continue
                if any(overlap((s, e), pi) > 0 for pi in ped_intervals):
                    events.append([round(s, 2), round(e, 2), "failure_to_yield"])
                    ids.append(oid)
    return (events, ids) if return_ids else events


# Пороги для accident/near_miss — те же единицы ("длины корпуса" и
# секунды), что уже калибровались (неформально, на глаз) для TTC-эвристики
# RiskEstimator в solution.py; здесь НЕ каузально (весь ролик разом), что
# позволяет смотреть и назад, и вперёд относительно момента сближения —
# в частности, отличать accident (после сближения объекты ОСТАНАВЛИВАЮТСЯ
# и стоят) от обычного плотного, но безопасного трафика (bbox тоже может
# перекрываться из-за перспективы камеры, особенно в очереди — separate
# STOP-признак нужен именно чтобы отсечь этот источник FP).
ACCIDENT_DIST_NORM = 0.35     # "длины корпуса" — боксы фактически перекрываются
DANGER_DIST_NORM = 1.6        # ближе этого — уже опасное сближение
TTC_DANGER_NONCAUSAL = 1.5    # сек
STILL_SPEED = 0.12
MOVING_SPEED = 0.25
STILL_HOLD_SEC = 1.5          # сколько нужно простоять, чтобы считать "остановкой из-за ДТП"
BRAKE_DROP_RATIO = 0.55       # относительное падение скорости для "резкого торможения"
PRE_CONTACT_SEC = 1.5         # окно ДО контакта, в котором кто-то из пары должен реально ехать
# near_miss (б) — одиночное резкое торможение без второго участника.
# ВЫКЛЮЧЕНО: каждая машина, тормозящая перед очередью на красный, даёт
# такое событие — на этой камере это сотни FP на ролик.
NEAR_MISS_FROM_BRAKING = False
PAIR_PREFILTER_NORM = 2.0     # пары, ни разу не сблизившиеся ближе — не рассматриваем


def _candidate_pairs(all_objs, thresh=PAIR_PREFILTER_NORM):
    """Пары объектов, которые хотя бы в одном кадре были ближе thresh
    "длин корпуса". Все объекты сэмплируются на одной сетке кадров, так что
    достаточно сравнить объекты внутри каждого кадра (numpy). Без этого
    перебор всех пар за весь ролик — O(N^2) по тысячам треков и
    десятки минут на одно видео."""
    by_t = {}
    for key, samples in all_objs.items():
        for s in samples:
            by_t.setdefault(s["t"], []).append((key, s["c"][0], s["c"][1], s["diag"]))
    pairs = set()
    for items in by_t.values():
        if len(items) < 2:
            continue
        keys = [it[0] for it in items]
        arr = np.array([it[1:] for it in items], dtype=np.float64)
        dx = arr[:, 0][:, None] - arr[:, 0][None, :]
        dy = arr[:, 1][:, None] - arr[:, 1][None, :]
        avg_diag = (arr[:, 2][:, None] + arr[:, 2][None, :]) / 2.0
        close = np.sqrt(dx ** 2 + dy ** 2) / avg_diag < thresh
        ii, jj = np.nonzero(np.triu(close, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            a, b = keys[i], keys[j]
            if a[0] == "p" and b[0] == "p":
                continue
            pairs.add((a, b) if a < b else (b, a))
    return sorted(pairs)


def _was_moving_before(samples, t0, window=PRE_CONTACT_SEC, min_speed=MOVING_SPEED):
    return any(t0 - window <= s["t"] <= t0 and s["speed"] >= min_speed for s in samples)


def _object_still_run_after(samples, t0, still_speed=STILL_SPEED, max_gap=2.0):
    """Возвращает (t_end_of_stop) если объект, начиная примерно с t0,
    непрерывно (с учётом max_gap) держит speed < still_speed минимум
    STILL_HOLD_SEC секунд; иначе None. t_end — момент, когда объект СНОВА
    начал двигаться (или последний сэмпл трека, если так и не поехал)."""
    still = [s for s in samples if s["t"] >= t0 - 1e-6]
    if not still:
        return None
    runs = sample_runs(still, lambda s: s["speed"] < still_speed, max_gap)
    for s, e in runs:
        if s <= t0 + 1.0 and e - s >= STILL_HOLD_SEC:  # остановка началась вскоре после t0
            return e
    return None


def _brake_events(samples, window_sec=2.0, min_speed=MOVING_SPEED, max_gap=1.5):
    """Интервалы резкого торможения без учёта другого объекта: скорость
    падает с >= min_speed до < min_speed*(1-BRAKE_DROP_RATIO) в пределах
    window_sec. Дешёвый одиночный сигнал (как _max_brake_ratio в
    RiskEstimator, но не каузально и на явном скользящем окне, а не на
    фиксированной истории из RISK_HISTORY сэмплов)."""
    events = []
    n = len(samples)
    for i in range(n):
        if samples[i]["speed"] < min_speed:
            continue
        for j in range(i + 1, n):
            dt = samples[j]["t"] - samples[i]["t"]
            if dt > window_sec:
                break
            if samples[j]["speed"] < min_speed * (1 - BRAKE_DROP_RATIO):
                events.append((samples[i]["t"], samples[j]["t"]))
                break
    return merge_intervals(events, gap=max_gap)


def detect_accidents_and_near_misses(vehicle_objs, person_objs, max_dt=0.6, return_ids=False):
    """ДТП и опасные сближения — единственные два класса, не завязанные
    на зоны камеры (чистая кинематика треков), но и единственные, где
    совсем нет готового учебного сигнала — это первая эвристическая
    попытка, требующая калибровки на реальной разметке.

    accident: пара объектов (машина-машина или машина-пешеход) сближается
    до dist_norm < ACCIDENT_DIST_NORM ("боксы фактически совместились") И
    после этого хотя бы один из них останавливается (speed < STILL_SPEED)
    минимум STILL_HOLD_SEC подряд. Именно требование "остановки после
    контакта" — попытка отсечь ложные срабатывания от простого визуального
    перекрытия боксов из-за перспективы (машины на разных полосах/разной
    глубине сцены, боксы которых пересекаются на 2D-кадре, но физически
    не соприкасаются) — если после "контакта" оба как ни в чём не бывало
    продолжают ехать, это, скорее всего, не авария, а перспективный
    артефакт трекинга.

    near_miss: либо (а) пара сблизилась до DANGER_DIST_NORM с закрытием
    достаточно быстрым, чтобы TTC < TTC_DANGER_NONCAUSAL, но БЕЗ контакта
    (dist_norm никогда не опускался ниже ACCIDENT_DIST_NORM) и затем разошлась,
    либо (б) отдельно взятый объект резко затормозил (_brake_events) не
    находясь под уже засчитанным accident. (а) специфичнее и вероятнее
    точен; (б) — грубый одиночный сигнал (резкое торможение бывает и без
    угрозы столкновения, например перед обычным красным) и даст больше
    FP — при калибровке через evaluate.py --per-video в первую очередь
    смотреть сюда, если near_miss ложно сработает слишком часто; при
    необходимости (б) можно просто выключить, оставив только (а)."""
    all_objs = {}
    for oid, samples in vehicle_objs.items():
        all_objs[("v", oid)] = samples
    for oid, samples in person_objs.items():
        all_objs[("p", oid)] = samples
    accidents, accident_pairs = [], []
    near_miss_candidates = []

    for ka, kb in _candidate_pairs(all_objs):
        sa, sb = all_objs[ka], all_objs[kb]
        approach = _closest_approach(sa, sb, max_dt=max_dt)
        if not approach:
            continue

        contact_runs = sample_runs(
            [{"t": t, "_ok": d < ACCIDENT_DIST_NORM} for t, d, _ in approach],
            lambda x: x["_ok"], max_gap=max_dt * 2,
        )
        had_accident = False
        for s, e in contact_runs:
            t_end_a = _object_still_run_after(sa, s)
            t_end_b = _object_still_run_after(sb, s)
            t_end = max(t_end_a or 0.0, t_end_b or 0.0)
            if t_end_a is None and t_end_b is None:
                continue  # контакт был, но оба продолжили ехать — вероятно, перспективный артефакт
            if not (_was_moving_before(sa, s) or _was_moving_before(sb, s)):
                continue  # оба стояли ещё до "контакта" — это соседи в очереди, чьи боксы
                          # перекрылись в перспективе, а не ДТП
            accidents.append([round(s, 2), round(max(t_end, e), 2), "accident"])
            accident_pairs.append((ka[1] if ka[0] == "v" else kb[1]))
            had_accident = True
        if had_accident:
            continue  # эта пара уже "потрачена" на accident — не дублируем near_miss

        min_dist, min_t = None, None
        for t, d, closing in approach:
            if d < DANGER_DIST_NORM and (min_dist is None or d < min_dist):
                min_dist, min_t = d, t
        if min_dist is not None and min_t is not None:
            # ищем момент вокруг минимума, где TTC был опасным
            danger_t = None
            for t, d, closing in approach:
                if closing and closing > 1e-6 and d / closing < TTC_DANGER_NONCAUSAL \
                        and abs(t - min_t) <= 3.0:
                    danger_t = t if danger_t is None else min(danger_t, t)
            if danger_t is not None:
                # окно near_miss: от начала опасного сближения до расхождения обратно за DANGER_DIST_NORM
                end_t = min_t
                for t, d, _ in approach:
                    if t >= min_t and d >= DANGER_DIST_NORM:
                        end_t = t
                        break
                else:
                    end_t = approach[-1][0]
                if end_t > danger_t:
                    near_miss_candidates.append((danger_t, end_t,
                                                  ka[1] if ka[0] == "v" else kb[1]))

    # (б) резкое торможение без привязки к конкретной другой машине — только
    # для объектов, ещё не отметившихся в accident выше в этом же интервале
    accident_intervals_by_obj = {}
    for (s, e, _), oid in zip(accidents, accident_pairs):
        accident_intervals_by_obj.setdefault(oid, []).append((s, e))
    for oid, samples in (vehicle_objs.items() if NEAR_MISS_FROM_BRAKING else ()):
        busy = merge_intervals(accident_intervals_by_obj.get(oid, []))
        for s, e in _brake_events(samples):
            if not any(overlap((s, e), b) > 0 for b in busy):
                near_miss_candidates.append((s, e, oid))

    # склеиваем near_miss кандидаты по объекту (та же машина, близкие интервалы)
    by_obj: dict = {}
    for s, e, oid in near_miss_candidates:
        by_obj.setdefault(oid, []).append((s, e))
    near_misses, nm_ids = [], []
    for oid, intervals in by_obj.items():
        for s, e in merge_intervals(intervals, gap=1.0):
            near_misses.append([round(s, 2), round(e, 2), "near_miss"])
            nm_ids.append(oid)

    if return_ids:
        return accidents, accident_pairs, near_misses, nm_ids
    return accidents, near_misses


def merge_same_class_overlaps(events):
    """Пересекающиеся сегменты одного класса — по FAQ задания это ОДНО
    событие ("два события одного класса одновременно -> один сегмент,
    покрывающий оба"), а не повод терять один из них.

    Раньше здесь было drop_same_class_overlaps, копирующее поведение
    run_submission.py (при пересечении внутри класса оставляет более
    ранний сегмент, остальные роняет как FP/дубликат) — как safety-net на
    стороне харнесса это разумно, но если мы САМИ так фильтруем перед
    отправкой, мы теряем recall каждый раз, когда две РАЗНЫЕ машины
    одновременно стоят/жгут красный: одно из двух настоящих событий
    молча исчезает вместо объединения в покрывающий сегмент."""
    by_label: dict[str, list[tuple[float, float]]] = {}
    for s, e, lbl in events:
        by_label.setdefault(lbl, []).append((s, e))
    out = []
    for lbl, intervals in by_label.items():
        for s, e in merge_intervals(intervals, gap=0.0):
            out.append([round(s, 2), round(e, 2), lbl])
    return sorted(out)


def compute_events(records, zones, light_samples=None, classes=None):
    """Единая точка входа без файлового I/O — её вызывает pipeline.infer().

    Args:
        records: список записей треков (уже со stitched_id, см.
            stitch_tracks.stitch_records()).
        zones: dict {zone_name: np.ndarray[[x,y],...]} — как из load_zones().
            Зоны "проезжей части" для jaywalking собираются по конвенции
            имени: 'roadway' и всё, что начинается с 'roadway_' или
            'crossroad' (roadway, roadway_before_queue, crossroad,
            crossroad_2, ...). Зоны легитимных переходов ('crossing_far',
            'crossing_near', любое 'crossing_*') автоматически исключаются
            из проезжей части, даже если геометрически пересекаются.
        light_samples: список [t_sec, state] из light_state.run_light_state(),
            или None если light_roi не размечен / решили не считать
            red_light/stop_line.
        classes: какие классы считать (None — все). Правила остальных классов
            не запускаются вовсе: на загруженном 10-минутном видео
            экспериментальные правила — десятки секунд внутри бюджета.

    Returns:
        Список [start_sec, end_sec, label] событий, уже без пересечений
        внутри класса.
    """
    events = compute_events_debug(records, zones, light_samples, classes)
    return merge_same_class_overlaps([e[:3] for e in events])


def _scene_zone_sets(zones):
    """Смысловые группы зон сцены (общие для compute_events и _debug):
      roadway    — проезжая часть (roadway*, crossroad*);
      ped_safe   — где пешеходу можно: переходы (crossing*) и тротуарные
                   островки (sidewalk*: они внутри полигона crossroad, и
                   люди, идущие с одной зебры на другую через островок,
                   давали большинство ложных jaywalking);
      road       — где стоящая машина мешает движению (stopped_vehicle):
                   проезжая часть + переходы + очередь + стоп-линия."""
    roadway = _zones_by_prefix(zones, ("roadway", "crossroad"))
    crossing = _zones_by_prefix(zones, ("crossing",))
    ped_safe = crossing | _zones_by_prefix(zones, ("sidewalk",))
    road = roadway | crossing | ({"queue_zone", "stop_line"} & set(zones))
    return roadway, ped_safe, road


def compute_events_debug(records, zones, light_samples=None, classes=None):
    """Все правила с привязкой к объекту. compute_events() — это же самое
    плюс merge_same_class_overlaps (одна реализация: сабмит и визуализация
    не могут разойтись).

    Отличия от compute_events():
      - события НЕ проходят через merge_same_class_overlaps: два разных
        нарушителя, пересекающиеся по времени, остаются двумя отдельными
        записями, а не одним покрывающим сегментом — иначе теряется
        привязка к конкретному объекту;
      - каждая запись — [start, end, label, object_id], где object_id это
        stitched_id виновника (см. stitch_tracks.py). Для "congestion"
        object_id всегда None: это событие про кластер из нескольких
        машин сразу, а не про одну — используйте zones['queue_zone'] +
        координаты боксов в этот момент, чтобы подсветить весь кластер.

    Возвращает список [start, end, label, object_id_or_None], отсортированный
    по времени начала.
    """
    groups = group_by_object(records)
    vehicle_objs = {oid: annotate(recs, zones) for oid, recs in groups.items()
                    if recs[0]["cls"] in VEHICLE_CLASSES}
    person_objs = {oid: annotate(recs, zones) for oid, recs in groups.items()
                   if recs[0]["cls"] == PERSON_CLASS}
    _mark_riders(groups, person_objs)

    roadway_zones, ped_safe_zones, road_zones = _scene_zone_sets(zones)
    crossing_zones = _zones_by_prefix(zones, ("crossing",))

    light = LightState(light_samples) if light_samples else None
    want = (lambda *labels: True) if classes is None else (lambda *labels: bool(set(labels) & set(classes)))
    out = []

    def run(label, fn, per_object=True):
        """Одно правило: только если его класс запрошен, и с собственной
        страховкой — падение экспериментального правила на незнакомой сцене
        не должно стирать остальные классы видео."""
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001 — логируем и продолжаем
            print(f"[rules] {label} упало: {exc!r}")
            return [], []
        events, ids = res if per_object else (res, [None] * len(res))
        out.extend([s, e, lbl, oid] for (s, e, lbl), oid in zip(events, ids))
        return events, ids

    if want("congestion"):
        run("congestion", lambda: detect_congestion(
            vehicle_objs, signal_zones={"queue_zone"} & set(zones) or {"queue_zone"},
            junction_zones=_zones_by_prefix(zones, ("crossroad",)), light=light), per_object=False)
    if want("stopped_vehicle"):
        run("stopped_vehicle", lambda: detect_stopped_vehicle(
            vehicle_objs, road_zones=road_zones, junction_zones=_zones_by_prefix(zones, ("crossroad",)),
            light=light, return_ids=True))
    if want("jaywalking"):
        run("jaywalking", lambda: detect_jaywalking(person_objs, roadway_zones, ped_safe_zones,
                                                    return_ids=True))

    if light is not None and "stop_line" in zones and want("red_light", "stop_line"):
        red, red_ids = [], []
        if "crossing_far" in zones:
            red, red_ids = run("red_light", lambda: detect_red_light(vehicle_objs, light, return_ids=True))
        if want("stop_line"):
            # stop_line — "встал за стоп-линией, НЕ въехав на перекрёсток"; кто потом
            # проехал на красный, тот red_light (C3902, 97 с: мотоцикл получал оба)
            red_set = set(red_ids)

            def stop_line():
                events, ids = detect_stop_line(vehicle_objs, light, return_ids=True)
                keep = [k for k, oid in enumerate(ids) if oid not in red_set]
                return [events[k] for k in keep], [ids[k] for k in keep]
            run("stop_line", stop_line)
        if not want("red_light"):
            out[:] = [ev for ev in out if ev[2] != "red_light"]

    if want("wrong_way"):
        run("wrong_way", lambda: detect_wrong_way(vehicle_objs, roadway_zones, return_ids=True))
    if want("illegal_u_turn"):
        run("illegal_u_turn", lambda: detect_illegal_u_turn(vehicle_objs, roadway_zones, return_ids=True))
    if want("illegal_turn"):
        run("illegal_turn", lambda: detect_illegal_turn(vehicle_objs, zones, return_ids=True))
    if want(CURB_MOUNT):
        run(CURB_MOUNT, lambda: detect_curb_mount(vehicle_objs, zones))
    if want("solid_line_crossing"):
        run("solid_line_crossing", lambda: detect_solid_line_crossing(vehicle_objs, zones, return_ids=True))
    if want("failure_to_yield"):
        run("failure_to_yield", lambda: detect_failure_to_yield(vehicle_objs, person_objs, crossing_zones,
                                                                return_ids=True))
    if want("accident", "near_miss"):
        def accidents():
            acc, _acc_ids, nm, nm_ids = detect_accidents_and_near_misses(
                vehicle_objs, person_objs, return_ids=True)
            # accident — про ДВА объекта сразу; object_id одного "виновника"
            # вводил бы в заблуждение при подсветке, поэтому None (как у congestion)
            return acc + nm, [None] * len(acc) + list(nm_ids)
        run("accident/near_miss", accidents)

    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------- main (dev only)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="*.stitched.json из stitch_tracks.py")
    ap.add_argument("--zones", required=True)
    ap.add_argument("--light", default=None, help="*.json из light_state.py (для red_light/stop_line)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta, records = load_tracks(args.tracks)
    zones = load_zones(args.zones)
    light_samples = None
    if args.light:
        light_samples = json.loads(Path(args.light).read_text())["samples"]

    events = compute_events(records, zones, light_samples)

    fps = meta.get("fps", 25.0)
    duration = meta.get("n_frames_total", 0) / fps if fps else 0.0

    out_path = Path(args.out) if args.out else Path("predictions") / (Path(meta["video"]).stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"video": meta["video"], "duration": round(duration, 2), "fps": fps, "events": events}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    by_class = {}
    for _, _, lbl in events:
        by_class[lbl] = by_class.get(lbl, 0) + 1
    print(f"{meta['video']}: {len(events)} событий -> {out_path}  {by_class}")


if __name__ == "__main__":
    main()