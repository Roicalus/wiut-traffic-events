"""risk.py — Part B: каузальная оценка P(accident начнётся в ближайшие 5 с).

Эвристика без обучения:
  1. YOLO11n + ByteTrack на каждом stride-м кадре (свой проход, каузальный —
     треки Part A сюда не попадают, они считались по всему видео);
  2. точка трека — низ-центр бокса (контакт с землёй: для машины и
     пешехода это одна плоскость, центр бокса высокой машины "висит"
     над соседней полосой); скорость — по смещению за 0.5 с, и ещё одна
     за предыдущие 0.5 с (видно, тормозит ли пара);
  3. для пар (хотя бы один участник — ТС) в единицах "диагональ бокса":
     точка наибольшего сближения при постоянной скорости (t*, d_min) и
     ТРЕБУЕМОЕ ЗАМЕДЛЕНИЕ a_req = v_c^2 / (2 * зазор), v_c — скорость
     сближения. Риск — только если пара идёт в контакт (d_min мал), скоро
     (t* мал) И остановиться уже трудно (a_req велико). Именно a_req
     отличает аварию от нормы: на сэмплах 80% ложных алармов старой версии
     — машина подъезжает к стоящей очереди, при постоянной скорости она
     "врежется" через 1 с, но скорость сближения мала и водитель тормозит;
  4. пара уже тормозит (скорость сближения упала) — риск снижается;
  5. мягкие сигналы ниже порога аларма (0.5), они только ранжируют кадры
     для AP: курс на столкновение без учёта a_req (SOFT_CAP) — появляется
     раньше, чем "остановиться уже трудно", и резкое торможение (BRAKE_CAP):
     само по себе торможение перед очередью на красный не даёт аларм,
     но поднимает ранжирование (AP);
  6. медиана последних MEDIAN_K сырых значений — гасит всплески короче ~0.3 с
     (перескоки ID и дрожание боксов в плотном потоке).

Пороги подобраны на сэмплах (tools/risk_replay.py): на обычном трафике
риск >= 0.5 — доли процента кадров, а синтетические сценарии столкновения
(tests/test_risk.py) дают аларм за 1-3 с до контакта.

Устройство: RiskEstimator = детектор (YOLO + ByteTrack, _detect) + бюджет
времени + RiskScorer (шаги 2-6, чистый numpy). RiskScorer не знает про
модель, поэтому tools/risk_replay.py кэширует детекции один раз и
перебирает пороги скоринга за секунды — с тем же кодом, что в сабмите.

Бюджет времени: step() знает дедлайн видео (src/budget.py) и, если не
успевает, увеличивает stride; в самом крайнем случае перестаёт вызывать
детектор совсем. Превышение бюджета обнулило бы и Part A этого видео.
"""
from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src import budget, track

RISK_MODEL = str(Path(__file__).resolve().parent.parent / "weights" / "yolo11n.pt")
RISK_TRACKER = "bytetrack.yaml"
RISK_IMGSZ = 640
RISK_CONF = 0.35
CROP_TOP_FRAC = 0.18       # как в Part A: верх кадра — деревья/небо
BASE_STRIDE = 2
MAX_STRIDE = 25            # реже раза в секунду смысла нет
REPLAN_WARMUP_FRAMES = 150  # разгон CUDA и первые декоды не в счёт
REPLAN_MIN_FRAMES = 300     # темп меряем хотя бы по 10 с видео
HISTORY = 30               # точек на трек: >= 1 с при BASE_STRIDE
STALE_SEC = 1.0
VEL_BASELINE_SEC = 0.5

# Все расстояния — в диагоналях бокса пары (перспектива), скорости — диаг/с.
# Для масштаба: машина в движении на сэмплах — медиана 0.7, 90% — 1.7 диаг/с.
HORIZON = 5.0              # дальше t* пару не рассматриваем
TTC_DANGER = 1.0           # t* <= этого -> временной множитель 1
TTC_SAFE = 3.5             # t* >= этого -> 0
D_COLL = 0.3               # d_min ниже — контакт (соседние полосы в перспективе — 0.5-0.8)
MAX_PAIR_DIST = 6.0        # дальше — пару не рассматриваем
VC_MIN = 0.8               # скорость сближения ниже — не опасно
GAP0 = 0.3                 # расстояние "точек на земле" в момент контакта
A_LOW, A_HIGH = 1.0, 3.0   # требуемое замедление, диаг/с^2: множитель 0 -> 1
BRAKING_RATIO = 0.75       # сближение упало ниже этой доли за 0.5 с = тормозят
BRAKING_FACTOR = 0.5
SOFT_CAP = 0.3             # "курс на столкновение" без учёта a_req: ниже порога аларма,
                           # но поднимает кадры за 1-5 с до контакта в ранжировании (AP)
MOVING_SPEED = 0.3
BORDER_PX = 8              # бокс у края кадра обрезан: его "точка на земле" скачет
BRAKE_LOW, BRAKE_HIGH = 0.4, 0.7
BRAKE_CAP = 0.35           # торможение само по себе не поднимает риск до аларма
MEDIAN_K = 9                # медиана 0.6 с при 15 Гц: на сэмплах 28 -> 4 ложных алармов, аларм позже на ~0.2 с

VEHICLE_CLS = {1, 2, 3, 5, 7}
PERSON_CLS = 0


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


class RiskScorer:
    """Риск по потоку детекций. feed(dets, t_sec) -> score; dets — массив
    (N, 6): x1, y1, x2, y2, track_id, cls. Только прошлое: состояние —
    истории треков и последние сырые значения. frame_wh — размер кадра, в
    котором заданы боксы (для отсева обрезанных краем кадра)."""

    def __init__(self, frame_wh: tuple[float, float] | None = None):
        self.frame_wh = frame_wh
        self.tracks: dict[int, deque] = {}
        self.raw_hist: deque = deque(maxlen=MEDIAN_K)
        self.score = 0.0
        self.explain: dict | None = None   # пара с наибольшим риском на последнем шаге

    def feed(self, dets: np.ndarray, t_sec: float) -> float:
        self._update_tracks(dets, t_sec)
        raw = max(self._pair_risk(t_sec), BRAKE_CAP * self._brake_risk())
        self.raw_hist.append(raw)
        self.score = round(float(np.median(self.raw_hist)), 4)
        return self.score

    def _update_tracks(self, dets, t_sec):
        if self.frame_wh is not None and len(dets):
            w, h = self.frame_wh
            inside = ((dets[:, 0] > BORDER_PX) & (dets[:, 1] > BORDER_PX)
                      & (dets[:, 2] < w - BORDER_PX) & (dets[:, 3] < h - BORDER_PX))
            dets = dets[inside]
        for x1, y1, x2, y2, tid, cls in dets:
            diag = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)
            self.tracks.setdefault(int(tid), deque(maxlen=HISTORY)).append(
                (t_sec, (x1 + x2) / 2.0, float(y2), diag, int(cls)))
        for tid in [k for k, d in self.tracks.items() if t_sec - d[-1][0] > STALE_SEC]:
            del self.tracks[tid]

    @staticmethod
    def _back(d, t, lag):
        """Последняя точка трека не позже t - lag."""
        for pt in reversed(d):
            if t - pt[0] >= lag:
                return pt
        return None

    def _state(self, d, t_sec):
        """(pos, vel, vel_prev|None, diag, cls) в px и px/с, или None, если
        трек не свежий или короче VEL_BASELINE_SEC."""
        t, x, y, diag, cls = d[-1]
        if t_sec - t > 1e-6:
            return None
        b = self._back(d, t, VEL_BASELINE_SEC)
        if b is None:
            return None
        vel = np.array([x - b[1], y - b[2]]) / (t - b[0])
        c = self._back(d, b[0], VEL_BASELINE_SEC)
        vel_prev = None if c is None else np.array([b[1] - c[1], b[2] - c[2]]) / (b[0] - c[0])
        return np.array([x, y]), vel, vel_prev, diag, cls

    def _pair_risk(self, t_sec) -> float:
        self.explain = None
        fresh = [(k, st) for k, st in ((k, self._state(d, t_sec)) for k, d in self.tracks.items()) if st]
        tids = [k for k, _ in fresh]
        states = [st for _, st in fresh]
        if len(states) < 2:
            return 0.0
        pos = np.array([s[0] for s in states])
        vel = np.array([s[1] for s in states])
        has_prev = np.array([s[2] is not None for s in states])
        vel_prev = np.array([s[1] if s[2] is None else s[2] for s in states])
        diag = np.array([s[3] for s in states])
        cls = np.array([s[4] for s in states])
        is_veh = np.isin(cls, list(VEHICLE_CLS))
        moving = np.linalg.norm(vel, axis=1) / diag >= MOVING_SPEED

        i, j = np.triu_indices(len(states), k=1)
        keep = (is_veh[i] | is_veh[j]) & (moving[i] | moving[j])
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return 0.0
        d_norm = ((diag[i] + diag[j]) / 2.0)[:, None]
        p = (pos[i] - pos[j]) / d_norm
        v = (vel[i] - vel[j]) / d_norm
        dist = np.linalg.norm(p, axis=1)
        v_c = -(p * v).sum(axis=1) / np.maximum(dist, 1e-6)        # > 0: сближаются
        ok = (dist < MAX_PAIR_DIST) & (v_c >= VC_MIN)
        if not ok.any():
            return 0.0
        i, j, p, v, dist, v_c = i[ok], j[ok], p[ok], v[ok], dist[ok], v_c[ok]
        vv = (v ** 2).sum(axis=1)
        t_star = -(p * v).sum(axis=1) / vv
        d_min = np.linalg.norm(p + v * t_star[:, None], axis=1)
        hit = (t_star > 0) & (t_star <= HORIZON) & (d_min < D_COLL)
        if not hit.any():
            return 0.0
        i, j, p, dist, v_c = i[hit], j[hit], p[hit], dist[hit], v_c[hit]
        t_star, d_min = t_star[hit], d_min[hit]

        a_req = v_c ** 2 / (2.0 * np.maximum(dist - GAP0, 0.05))
        r_time = np.clip((TTC_SAFE - t_star) / (TTC_SAFE - TTC_DANGER), 0.0, 1.0)
        r_acc = np.clip((a_req - A_LOW) / (A_HIGH - A_LOW), 0.0, 1.0)
        r_geom = 1.0 - 0.5 * d_min / D_COLL
        # сближение полсекунды назад: если сейчас заметно медленнее — пара тормозит
        vp = (vel_prev[i] - vel_prev[j]) / ((diag[i] + diag[j]) / 2.0)[:, None]
        v_c_prev = -(p * vp).sum(axis=1) / np.maximum(dist, 1e-6)
        braking = has_prev[i] & has_prev[j] & (v_c < BRAKING_RATIO * v_c_prev)
        r = r_time * r_acc * r_geom * np.where(braking, BRAKING_FACTOR, 1.0)
        r = np.maximum(r, SOFT_CAP * np.clip((HORIZON - t_star) / (HORIZON - TTC_DANGER), 0.0, 1.0) * r_geom)
        k = int(r.argmax())
        self.explain = {"ids": (tids[i[k]], tids[j[k]]), "cls": (int(cls[i[k]]), int(cls[j[k]])),
                        "pos": (pos[i[k]].round(0).tolist(), pos[j[k]].round(0).tolist()),
                        "dist": round(float(dist[k]), 2), "v_c": round(float(v_c[k]), 2),
                        "t_star": round(float(t_star[k]), 2), "d_min": round(float(d_min[k]), 2),
                        "a_req": round(float(a_req[k]), 2), "braking": bool(braking[k]),
                        "v_c_prev": round(float(v_c_prev[k]), 2),
                        "diag": (round(float(diag[i[k]])), round(float(diag[j[k]]))),
                        "risk": round(float(r[k]), 3)}
        return float(r[k])

    def _brake_risk(self) -> float:
        best = 0.0
        for d in self.tracks.values():
            if len(d) < 6 or d[-1][4] not in VEHICLE_CLS:
                continue
            pts = list(d)
            mid = len(pts) // 2

            def speed(a, b):
                dt = b[0] - a[0]
                if dt <= 0:
                    return None
                return np.hypot(b[1] - a[1], b[2] - a[2]) / ((a[3] + b[3]) / 2.0) / dt

            v_before, v_after = speed(pts[0], pts[mid]), speed(pts[mid], pts[-1])
            if v_before and v_after is not None and v_before >= MOVING_SPEED:
                best = max(best, (v_before - v_after) / v_before)
        return _clip01((best - BRAKE_LOW) / (BRAKE_HIGH - BRAKE_LOW))


class RiskEstimator:
    """Детектор на каждом stride-м кадре + RiskScorer.

    Детекция идёт в фоновом потоке, пока харнесс декодирует следующие кадры
    (декодирование 4K — главный расход времени, и оно не ждёт GPU). Детекция
    кадра k запускается в step(k), а её результат забирается строго в
    следующем step() с инференсом — поэтому результат не зависит от скорости
    потоков (детерминизм), а скор в момент t считается только по кадрам до t
    (причинность). Цена — задержка на один шаг: stride кадров, ~0.07 с.
    """
    _model = None  # веса грузим один раз на процесс
    _pool = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from ultralytics import YOLO
            cls._model = YOLO(RISK_MODEL)
            cls._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="risk-det")
        return cls._model

    # ------------------------------------------------------------ API
    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.fps = float(meta.get("fps") or 25.0)
        self.n_frames = int(meta.get("n_frames") or 0)
        duration = self.n_frames / self.fps if self.fps else 0.0
        self.deadline = budget.deadline_for(meta.get("video_id", ""), duration)
        self.model = self._get_model()
        self._drain()                      # хвост предыдущего видео, если был
        self.device = track._pick_device()
        self.half = self.device != "cpu"
        width, height = int(meta.get("width") or 0), int(meta.get("height") or 0)
        crop_h = height - int(height * CROP_TOP_FRAC)
        self.scorer = RiskScorer((width, crop_h) if width and height else None)
        self.last_score = 0.0
        self.idx = -1
        self.stride = BASE_STRIDE
        self.disabled = False
        self.first_call = True
        self._pending = None               # (future, t_sec) детекции в работе
        self.n_failed = 0
        self.explain_log = None            # демо: список (t, score, пара риска) для отрисовки
        self._base = None                  # (wall, idx): с какого места меряем темп
        self._over = 0                     # сколько проверок подряд прогноз за дедлайном

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.idx += 1
        if self.idx % 50 == 0:
            self._replan(time.perf_counter())
        if self.disabled or self.idx % self.stride != 0:
            return self.last_score
        self._collect()
        self._pending = (self._pool.submit(self._detect, frame), t_sec)
        return self.last_score

    # ------------------------------------------------------------ детекция
    def _collect(self) -> None:
        """Забирает детекцию, запущенную в прошлом шаге, и обновляет скор."""
        if self._pending is None:
            return
        fut, t_det = self._pending
        self._pending = None
        try:
            dets = fut.result()
        except Exception as exc:  # один плохой кадр не должен ронять весь прогон
            self.n_failed += 1
            if self.n_failed == 1:
                print(f"[risk] детектор упал на t={t_det:.1f}s: {exc!r} (дальше — без повторов в логе)")
            return
        self.last_score = self.scorer.feed(dets, t_det)
        if self.explain_log is not None:   # только визуализация (демо); на скор не влияет
            self.explain_log.append((t_det, self.last_score, self.scorer.explain))

    def _drain(self) -> None:
        pending = getattr(self, "_pending", None)
        if pending is not None:
            try:
                pending[0].result()
            except Exception:
                pass
        self._pending = None

    def _detect(self, frame) -> np.ndarray:
        """(N, 6) float32: x1, y1, x2, y2, track_id, cls в координатах обрезанного кадра."""
        y0 = int(frame.shape[0] * CROP_TOP_FRAC)
        r = self.model.track(frame[y0:], persist=not self.first_call, tracker=RISK_TRACKER,
                             classes=track.CLASSES_OF_INTEREST, imgsz=RISK_IMGSZ,
                             conf=RISK_CONF, device=self.device, half=self.half,
                             verbose=False)[0]
        self.first_call = False
        if r.boxes is None or r.boxes.id is None:
            return np.zeros((0, 6), np.float32)
        return np.column_stack([r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy(),
                                r.boxes.cls.cpu().numpy()]).astype(np.float32)

    # ------------------------------------------------------------ бюджет
    def _replan(self, now: float) -> None:
        """Прогноз конца видео по СРЕДНЕМУ темпу (декод харнесса + наш инференс)
        с момента после разгона. Не успеваем — реже детектор; дедлайн прошёл —
        детектор выключен и скор 0 (старый скор стал бы одним длинным алармом).

        Темп по последним 50 кадрам был слишком нервным: одна медленная секунда
        декодирования давала прогноз "+19 s к дедлайну" там, где видео кончилось
        с запасом 300 с (C3902), stride рос, и кривая риска зависела от того,
        что ещё грузило машину. Теперь: среднее с начала, и две проверки подряд
        за дедлайном — только тогда stride += 1 (и темп меряется заново)."""
        if self.deadline == float("inf") or self.n_frames <= 0:
            return
        if now > self.deadline:
            if not self.disabled:
                print(f"[risk] бюджет исчерпан на t={self.idx / self.fps:.0f}s — детектор выключен")
            self.disabled = True
            self._drain()
            self.last_score = 0.0
            return
        if self.idx < REPLAN_WARMUP_FRAMES:        # первые кадры — разгон CUDA
            return
        if self._base is None:
            self._base = (now, self.idx)
            return
        done = self.idx - self._base[1]
        if done < REPLAN_MIN_FRAMES:
            return
        rate = (now - self._base[0]) / done
        projected = now + rate * max(self.n_frames - self.idx, 0)
        self._over = self._over + 1 if projected > self.deadline else 0
        if self._over >= 2 and self.stride < MAX_STRIDE:
            self.stride += 1
            self._base, self._over = (now, self.idx), 0
            print(f"[risk] прогноз {projected - self.deadline:+.0f}s к дедлайну -> stride={self.stride}")
