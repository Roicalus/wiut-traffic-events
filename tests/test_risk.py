"""Part B: синтетические сценарии для RiskScorer (без GPU).

Кадры подаются с частотой инференса (30 fps, BASE_STRIDE=2 -> 15 Гц).
Машина — бокс 300x160 px (диагональ 340), пешеход — 60x170 (180);
скорости в диагоналях/с, как в src/risk.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk import RiskScorer  # noqa: E402

DT = 2 / 30.0
CAR, PED = (300.0, 160.0), (60.0, 170.0)
CAR_DIAG = float(np.hypot(*CAR))


def box(tid, cls, ground_xy, size):
    x, y = ground_xy                  # низ-центр бокса
    w, h = size
    return [x - w / 2, y - h, x + w / 2, y, tid, cls]


def run(objects, t_end):
    """objects: список (tid, cls, size, f(t) -> ground_xy | None).
    Возвращает [(t, score)]."""
    sc, out = RiskScorer(), []
    for t in np.arange(0.0, t_end, DT):
        dets = [box(tid, cls, f(t), size) for tid, cls, size, f in objects if f(t) is not None]
        out.append((t, sc.feed(np.array(dets, np.float32).reshape(-1, 6), float(t))))
    return out


def peak_before(curve, t_contact, window=3.0):
    return max(s for t, s in curve if t_contact - window <= t < t_contact)


def test_t_bone_crossing_alarms_before_contact():
    v = 2.0 * CAR_DIAG                              # ~35 км/ч
    meet, t_c = np.array([2000.0, 1200.0]), 3.0
    a = lambda t: (meet[0] - v * (t_c - t), meet[1])
    b = lambda t: (meet[0], meet[1] - v * (t_c - t))
    curve = run([(1, 2, CAR, a), (2, 2, CAR, b)], t_c - 0.3)
    assert peak_before(curve, t_c) >= 0.5
    # аларм, только когда "остановиться уже трудно" (a_req): поздно, зато без
    # ложных на обычном трафике; раньше кадры поднимает мягкий сигнал SOFT_CAP
    first = min(t for t, s in curve if s >= 0.5)
    assert t_c - first >= 0.5, f"аларм всего за {t_c - first:.2f} с"
    assert max(s for t, s in curve if t_c - 3.0 <= t < t_c - 1.5) > 0.1


def test_rear_end_into_stopped_car_alarms():
    v, gap0 = 2.5 * CAR_DIAG, 7.0 * CAR_DIAG
    stopped = lambda t: (2000.0, 1200.0)
    follower = lambda t: (2000.0 - gap0 + v * t, 1200.0)
    t_c = (gap0 - 0.5 * CAR_DIAG) / v
    curve = run([(1, 2, CAR, stopped), (2, 2, CAR, follower)], t_c - 0.2)
    assert peak_before(curve, t_c) >= 0.5


def test_pedestrian_steps_in_front_of_car_alarms():
    v_car, v_ped, t_c = 2.0 * CAR_DIAG, 200.0, 3.0
    hit = np.array([2000.0, 1200.0])
    car = lambda t: (hit[0] - v_car * (t_c - t), hit[1])
    ped = lambda t: (hit[0], hit[1] - v_ped * (t_c - t))
    curve = run([(1, 2, CAR, car), (2, 0, PED, ped)], t_c - 0.2)
    assert peak_before(curve, t_c) >= 0.5


def test_braking_into_queue_is_quiet():
    """1.5 диаг/с, плавное торможение до полной остановки за корпус до очереди."""
    v0, stop_gap = 1.5 * CAR_DIAG, 1.2 * CAR_DIAG
    x_queue = 2500.0
    decel = 0.5 * CAR_DIAG                           # ~2.5 м/с^2
    t_stop = v0 / decel
    x0 = x_queue - stop_gap - v0 * t_stop / 2

    def follower(t):
        tt = min(t, t_stop)
        return (x0 + v0 * tt - decel * tt ** 2 / 2, 1200.0)
    curve = run([(1, 2, CAR, lambda t: (x_queue, 1200.0)), (2, 2, CAR, follower)], t_stop + 2)
    assert max(s for _, s in curve) < 0.5


def test_parallel_lanes_and_oncoming_pass_are_quiet():
    lane = 1.5 * CAR_DIAG
    fast = lambda t: (500.0 + 2.5 * CAR_DIAG * t, 1200.0)
    slow = lambda t: (1500.0 + 1.0 * CAR_DIAG * t, 1200.0 + lane)
    oncoming = lambda t: (4000.0 - 2.0 * CAR_DIAG * t, 1200.0 - lane)
    curve = run([(1, 2, CAR, fast), (2, 2, CAR, slow), (3, 2, CAR, oncoming)], 5.0)
    assert max(s for _, s in curve) < 0.5


def test_stale_tracks_are_forgotten():
    sc = RiskScorer()
    sc.feed(np.array([box(1, 2, (100, 100), CAR)], np.float32), 0.0)
    sc.feed(np.zeros((0, 6), np.float32), 2.0)
    assert sc.tracks == {}


def _guard(n_frames=9000, deadline=600.0):
    from src.risk import RiskEstimator, BASE_STRIDE
    est = RiskEstimator.__new__(RiskEstimator)      # без модели: проверяем только бюджет
    est.deadline, est.n_frames, est.fps = deadline, n_frames, 30.0
    est.stride, est.disabled, est._base, est._over, est.idx = BASE_STRIDE, False, None, 0, 0
    est._pending, est.last_score = None, 0.0
    return est


def test_one_slow_second_does_not_thin_part_b():
    """Одна медленная секунда (декод, GC) не должна поднимать stride: иначе кривая
    риска зависит от нагрузки на машину (было на C3902, прогноз +19 s)."""
    est, now = _guard(), 0.0
    for idx in range(0, 9000, 50):
        now += 50 * (0.03 if not 3000 <= idx < 3050 else 1.0)   # 0.03 с/кадр, один провал
        est.idx = idx
        est._replan(now)
    assert est.stride == 2


def test_consistently_slow_run_raises_stride():
    est, now = _guard(deadline=300.0), 0.0
    for idx in range(0, 3000, 50):
        now += 50 * 0.06                                            # 540 с на 9000 кадров > 300
        est.idx = idx
        est._replan(now)
    assert est.stride > 2
