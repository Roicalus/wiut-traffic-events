"""risk_scenarios.py — проверка Part B (RiskScorer) на синтетических траекториях.

Сценарии с известным исходом: столкновение (контакт = рамки начинают
пересекаться) или его нет. Детекции подаются с частотой инференса (15 Гц),
с дрожанием рамок и пропусками детекций. Для каждого сценария — как в
evaluate.py: есть ли аларм (score >= 0.5) с началом в [s - 10, s), время
от начала аларма до контакта (TTA), пик риска; для безопасных — пик риска
и число алармов (все ложные).

    python tools/risk_scenarios.py
    python tools/risk_scenarios.py --set A_HIGH=2.0 --set BRAKING_FACTOR=0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import risk  # noqa: E402

DT = 2 / 30.0                        # BASE_STRIDE = 2 при 30 fps
CAR, PED, BIKE = (300.0, 160.0), (60.0, 170.0), (110.0, 150.0)
D = float(np.hypot(*CAR))            # диагональ машины, px; скорости — в D/с
THETA, W = 0.5, 10.0


def box(tid, cls, g, size):
    (x, y), (w, h) = g, size
    return [x - w / 2, y - h, x + w / 2, y, tid, cls]


def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def simulate(objects, t_end, seed=0, jitter=3.0, drop=0.1):
    """objects: (tid, cls, size, f(t) -> ground (x, y)). -> (t, score), t_contact|None."""
    rng = np.random.default_rng(seed)
    sc, curve, t_contact = risk.RiskScorer((3840, 1771)), [], None
    for t in np.arange(0.0, t_end, DT):
        truth = [box(tid, cls, f(t), size) for tid, cls, size, f in objects]
        if t_contact is None and any(overlap(truth[a], truth[b]) for a in range(len(truth))
                                     for b in range(a + 1, len(truth))):
            t_contact = float(t)
        dets = []
        for b in truth:
            if rng.random() < drop:
                continue
            b = list(b)
            b[:4] = [c + rng.normal(0, jitter) for c in b[:4]]
            dets.append(b)
        curve.append((float(t), sc.feed(np.array(dets, np.float32).reshape(-1, 6), float(t))))
    return curve, t_contact


def alarms(curve):
    runs = []
    for t, s in curve:
        if s < THETA:
            continue
        if runs and t - runs[-1][1] < 2.0:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    return runs


def lin(p0, v, t0=0.0):
    """Равномерно: из p0 со скоростью v (px/с), стартует в t0."""
    return lambda t: (p0[0] + v[0] * max(t - t0, 0), p0[1] + v[1] * max(t - t0, 0))


def braking(p0, v, t_brake, decel):
    """Едет со скоростью v, с t_brake тормозит с замедлением decel (px/с^2) до нуля."""
    sp = float(np.hypot(*v))
    u = (v[0] / sp, v[1] / sp)
    t_stop = sp / decel

    def f(t):
        if t <= t_brake:
            s = sp * t
        else:
            tb = min(t - t_brake, t_stop)
            s = sp * t_brake + sp * tb - decel * tb ** 2 / 2
        return (p0[0] + u[0] * s, p0[1] + u[1] * s)
    return f


def scenarios():
    """(name, crash?, objects, t_end)."""
    M = (2000.0, 1300.0)                          # точка встречи
    out = []
    for v in (1.0, 1.5, 2.0, 3.0):                # D/с; 1 D/с ~ 18 км/ч
        # наезд сзади на стоящую машину без торможения
        out.append((f"rear-end, {v} D/s, no braking", True,
                    [(1, 2, CAR, lambda t: M), (2, 2, CAR, lin((M[0] - 8 * D, M[1]), (v * D, 0)))], 12))
        # удар сбоку на перекрёстке
        tc = 4.0
        out.append((f"T-bone, {v} D/s", True,
                    [(1, 2, CAR, lin((M[0] - v * D * tc, M[1]), (v * D, 0))),
                     (2, 2, CAR, lin((M[0], M[1] - v * D * tc), (0, v * D)))], tc + 1))
        # пешеход выходит перед машиной
        out.append((f"pedestrian hit, car {v} D/s", True,
                    [(1, 2, CAR, lin((M[0] - v * D * tc, M[1]), (v * D, 0))),
                     (2, 0, PED, lin((M[0], M[1] - 180 * tc), (0, 180)))], tc + 1))
    # в последний момент тормозит, но не успевает (частый сценарий реальной аварии)
    for v in (2.0, 3.0):
        x0 = M[0] - 8 * D
        t_brake = (8 * D - 2.2 * D) / (v * D)       # увидел за ~2 корпуса
        out.append((f"rear-end, {v} D/s, late braking", True,
                    [(1, 2, CAR, lambda t: M), (2, 2, CAR, braking((x0, M[1]), (v * D, 0), t_brake, 0.6 * D))], 12))
    # мотоцикл в бок машине
    out.append(("motorcycle T-bone, 2 D/s", True,
                [(1, 2, CAR, lin((M[0] - 2 * D * 4, M[1]), (2 * D, 0))),
                 (2, 3, BIKE, lin((M[0], M[1] - 2 * D * 4), (0, 2 * D)))], 5))
    # --- безопасные
    x_q = M[0]
    for v in (1.0, 1.5, 2.0):
        dec = 0.5 * D                                 # ~2.5 м/с^2, обычное торможение
        t_stop = v * D / dec
        stop_gap = 1.2 * D
        x0 = x_q - stop_gap - v * D * t_stop / 2 - 1.0 * v * D
        out.append((f"brakes smoothly behind queue, {v} D/s", False,
                    [(1, 2, CAR, lambda t: (x_q, M[1])), (2, 2, CAR, braking((x0, M[1]), (v * D, 0), 1.0, dec))], 10))
    out.append(("near miss: crossing 1.5 s apart", False,
                [(1, 2, CAR, lin((M[0] - 2 * D * 4, M[1]), (2 * D, 0))),
                 (2, 2, CAR, lin((M[0], M[1] - 2 * D * 5.5), (0, 2 * D)))], 8))
    out.append(("parallel lanes, same speed", False,
                [(1, 2, CAR, lin((500, M[1]), (2 * D, 0))), (2, 2, CAR, lin((600, M[1] + 0.9 * D), (2 * D, 0)))], 8))
    out.append(("overtaking in the next lane", False,
                [(1, 2, CAR, lin((500, M[1]), (1 * D, 0))), (2, 2, CAR, lin((200, M[1] + 0.9 * D), (2.5 * D, 0)))], 8))
    out.append(("oncoming traffic passes", False,
                [(1, 2, CAR, lin((500, M[1]), (2 * D, 0))), (2, 2, CAR, lin((3500, M[1] + 1.0 * D), (-2 * D, 0)))], 8))
    out.append(("car passes behind a walking pedestrian", False,
                [(1, 2, CAR, lin((M[0] - 2 * D * 4, M[1]), (2 * D, 0))),
                 (2, 0, PED, lin((M[0], M[1] - 180 * 4 + 1.5 * D), (0, 180)))], 6))
    out.append(("pedestrians walk past each other", False,
                [(1, 0, PED, lin((1000, M[1]), (180, 0))), (2, 0, PED, lin((2500, M[1]), (-180, 0)))], 8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], help="NAME=VALUE: параметр src/risk.py")
    ap.add_argument("--seeds", type=int, default=5, help="прогонов шума на сценарий")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=")
        setattr(risk, k, type(getattr(risk, k))(v))

    hits = n_crash = false_alarms = 0
    ttas = []
    print(f"{'scenario':44s} {'outcome':9s} {'detected':>8s} {'TTA, s':>7s} {'peak':>5s}")
    for name, crash, objs, t_end in scenarios():
        det, tta, peak, fa = 0, [], [], 0
        for seed in range(args.seeds):
            curve, s = simulate(objs, t_end, seed=seed)
            if crash and s is None:
                raise SystemExit(f"{name}: контакта нет — ошибка сценария")
            before = [(t, x) for t, x in curve if s is None or t < s]
            peak.append(max(x for _, x in before))
            runs = alarms(before)
            if crash:
                m = [r for r in runs if s - W <= r[0] < s]
                if m:
                    det += 1
                    tta.append(s - m[0][0])
                fa += len(runs) - (1 if m else 0)
            else:
                fa += len(runs)
        false_alarms += fa
        if crash:
            n_crash += args.seeds
            hits += det
            ttas += tta + [0.0] * (args.seeds - det)
        print(f"{name:44s} {'CRASH' if crash else 'safe':9s} {f'{det}/{args.seeds}' if crash else '-':>8s} "
              f"{(f'{np.mean(tta):.2f}' if tta else '-'):>7s} {np.mean(peak):5.2f}"
              + (f"  false alarms: {fa}" if fa else ""))
    print(f"\nrecall {hits}/{n_crash}, mean TTA {np.mean(ttas):.2f} s (W = {W:g}), false alarms {false_alarms}")


if __name__ == "__main__":
    main()
