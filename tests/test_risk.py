"""Part B: synthetic scenarios for RiskScorer (no GPU).

Frames are fed at the inference rate (30 fps, BASE_STRIDE=2 -> 15 Hz).
A car is a 300x160 px box (diagonal 340), a pedestrian 60x170 (180);
speeds are in diagonals/s, as in src/risk.py.
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
    x, y = ground_xy                  # bottom-centre of the box
    w, h = size
    return [x - w / 2, y - h, x + w / 2, y, tid, cls]


def run(objects, t_end):
    """objects: list of (tid, cls, size, f(t) -> ground_xy | None).
    Returns [(t, score)]."""
    sc, out = RiskScorer(), []
    for t in np.arange(0.0, t_end, DT):
        dets = [box(tid, cls, f(t), size) for tid, cls, size, f in objects if f(t) is not None]
        out.append((t, sc.feed(np.array(dets, np.float32).reshape(-1, 6), float(t))))
    return out


def peak_before(curve, t_contact, window=3.0):
    return max(s for t, s in curve if t_contact - window <= t < t_contact)


def test_t_bone_crossing_alarms_before_contact():
    v = 2.0 * CAR_DIAG                              # ~35 km/h
    meet, t_c = np.array([2000.0, 1200.0]), 3.0
    a = lambda t: (meet[0] - v * (t_c - t), meet[1])
    b = lambda t: (meet[0], meet[1] - v * (t_c - t))
    curve = run([(1, 2, CAR, a), (2, 2, CAR, b)], t_c - 0.3)
    assert peak_before(curve, t_c) >= 0.5
    # alarm only once "stopping is already hard" (a_req): late, but with no
    # false alarms in normal traffic; earlier frames are raised by the soft SOFT_CAP signal
    first = min(t for t, s in curve if s >= 0.5)
    assert t_c - first >= 0.5, f"alarm only {t_c - first:.2f} s ahead"
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
    """1.5 diag/s, smooth braking to a full stop one car length before the queue."""
    v0, stop_gap = 1.5 * CAR_DIAG, 1.2 * CAR_DIAG
    x_queue = 2500.0
    decel = 0.5 * CAR_DIAG                           # ~2.5 m/s^2
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
    est = RiskEstimator.__new__(RiskEstimator)      # no model: only the budget is tested
    est.deadline, est.n_frames, est.fps = deadline, n_frames, 30.0
    est.stride, est.disabled, est._base, est._over, est.idx = BASE_STRIDE, False, None, 0, 0
    est._pending, est.last_score = None, 0.0
    return est


def test_one_slow_second_does_not_thin_part_b():
    """One slow second (decode, GC) must not raise the stride: otherwise the risk
    curve depends on machine load (happened on C3902, forecast +19 s)."""
    est, now = _guard(), 0.0
    for idx in range(0, 9000, 50):
        now += 50 * (0.03 if not 3000 <= idx < 3050 else 1.0)   # 0.03 s/frame, one dip
        est.idx = idx
        est._replan(now)
    assert est.stride == 2


def test_consistently_slow_run_raises_stride():
    est, now = _guard(deadline=300.0), 0.0
    for idx in range(0, 3000, 50):
        now += 50 * 0.06                                            # 540 s for 9000 frames > 300
        est.idx = idx
        est._replan(now)
    assert est.stride > 2


def test_track_id_handed_from_person_to_car_does_not_spike_risk():
    """A pedestrian's id passed to a car 5 m away from them: without resetting the history this is
    a "speed" of hundreds of px/s and a false risk."""
    from src.risk import RiskScorer
    sc = RiskScorer((3840, 1800))
    scores = []
    for k in range(60):
        t = k / 15
        dets = [[1000, 900, 1300, 1100, 2, 2]]                                    # stationary car
        if t < 2:
            dets.append([1500, 800, 1540, 900, 7, 0])                             # standing person
        else:
            dets.append([1320, 900, 1620, 1100, 7, 2])                            # same id — the neighbouring car
        scores.append(sc.feed(np.array(dets, np.float32), t))
    assert max(scores) < 0.5, max(scores)


def test_near_car_heading_for_a_car_far_down_the_road_is_not_a_conflict():
    """Same class, very different box sizes = different depths (C3902, 0:13.5): the image-space
    closest approach says 'contact in 0.5 s', but the far car is metres away on the ground."""
    from src.risk import RiskScorer
    sc = RiskScorer((3840, 2074))
    scores = []
    for k in range(40):
        t = k / 15
        near = [2900 - 600 * t - 180, 900 - 150 * t - 150, 2900 - 600 * t + 180, 900 - 150 * t, 1, 2]  # diag ~360
        far = [2320, 700, 2410, 740, 2, 2]                                                            # diag ~100, parked
        scores.append(sc.feed(np.array([near, far], np.float32), t))
    assert max(scores) < 0.5, max(scores)
