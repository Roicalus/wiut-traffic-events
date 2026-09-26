"""risk.py — Part B: causal estimate of P(an accident starts within the next 5 s).

Training-free heuristic:
  1. YOLO11n + ByteTrack on every stride-th frame (its own causal pass —
     Part A tracks are not used here, they were computed over the whole video);
  2. the track point is the bottom-centre of the box (ground contact: for a car and
     a pedestrian this is the same plane, while the box centre of a tall vehicle "hangs"
     over the neighbouring lane); velocity is the displacement over 0.5 s, plus one more
     over the preceding 0.5 s (shows whether the pair is braking);
  3. for pairs (at least one participant is a vehicle), in units of "box diagonal":
     the closest point of approach at constant velocity (t*, d_min) and the
     REQUIRED DECELERATION a_req = v_c^2 / (2 * gap), v_c being the closing
     speed. Risk only if the pair is heading for contact (small d_min), soon
     (small t*) AND stopping is already hard (large a_req). It is a_req that
     separates an accident from normal traffic: on the samples 80% of the old version's
     false alarms were a car approaching a stationary queue — at constant velocity it would
     "crash" in 1 s, but the closing speed is low and the driver brakes;
  4. the pair is already braking (closing speed dropped) — risk is reduced;
  5. soft signals below the alarm threshold (0.5) only rank frames
     for AP: collision course ignoring a_req (SOFT_CAP) — appears
     earlier than "stopping is already hard", and hard braking (BRAKE_CAP):
     braking alone before a queue at a red light does not raise an alarm,
     but lifts the ranking (AP);
  6. median of the last MEDIAN_K raw values — suppresses spikes shorter than ~0.3 s
     (ID switches and box jitter in dense traffic).

Thresholds were tuned on the samples (tools/risk_replay.py): on normal traffic
risk >= 0.5 occurs in a fraction of a percent of frames, while synthetic collision scenarios
(tests/test_risk.py) raise the alarm 1-3 s before contact.

Structure: RiskEstimator = detector (YOLO + ByteTrack, _detect) + time
budget + RiskScorer (steps 2-6, pure numpy). RiskScorer knows nothing about the
model, so tools/risk_replay.py caches detections once and
sweeps scoring thresholds in seconds — with the same code as in the submission.

Time budget: step() knows the video deadline (src/budget.py) and, if it is
falling behind, increases the stride; in the most extreme case it stops calling
the detector altogether. Exceeding the budget would also zero out Part A for this video.
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
CROP_TOP_FRAC = track.CROP_TOP_FRAC   # as in Part A: only the strip above the scene zones
BASE_STRIDE = 2
MAX_STRIDE = 25            # less often than once per second makes no sense
REPLAN_WARMUP_FRAMES = 150  # CUDA warm-up and the first decodes do not count
REPLAN_MIN_FRAMES = 300     # measure the pace over at least 10 s of video
HISTORY = 30               # points per track: >= 1 s at BASE_STRIDE
STALE_SEC = 1.0
VEL_BASELINE_SEC = 0.5

# All distances are in box diagonals of the pair (perspective), speeds in diag/s.
# For scale: a moving car on the samples — median 0.7, 90th percentile 1.7 diag/s.
HORIZON = 5.0              # pairs with t* beyond this are ignored
TTC_DANGER = 1.0           # t* <= this -> time factor 1
TTC_SAFE = 3.5             # t* >= this -> 0
D_COLL = 0.3               # d_min below this is contact (adjacent lanes in perspective are 0.5-0.8)
MAX_PAIR_DIST = 6.0        # pairs farther apart are ignored
VC_MIN = 0.8               # closing speed below this is not dangerous
GAP0 = 0.3                 # distance between the "ground points" at the moment of contact
A_LOW, A_HIGH = 1.0, 3.0   # required deceleration, diag/s^2: factor 0 -> 1
BRAKING_RATIO = 0.75       # closing speed fell below this fraction within 0.5 s = braking
BRAKING_FACTOR = 0.5
SOFT_CAP = 0.3             # "collision course" ignoring a_req: below the alarm threshold,
                           # but lifts frames 1-5 s before contact in the ranking (AP)
MOVING_SPEED = 0.3
BORDER_PX = 8              # a box at the frame edge is clipped: its "ground point" jumps
BRAKE_LOW, BRAKE_HIGH = 0.4, 0.7
BRAKE_CAP = 0.35           # braking alone does not raise the risk to alarm level
MEDIAN_K = 9                # 0.6 s median at 15 Hz: on the samples 28 -> 4 false alarms, alarm ~0.2 s later

SAME_CLASS_MAX_SIZE_RATIO = 2.2  # two cars whose boxes differ more are at different depths

VEHICLE_CLS = {1, 2, 3, 5, 7}
PERSON_CLS = 0


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


class RiskScorer:
    """Risk from a stream of detections. feed(dets, t_sec) -> score; dets is an array
    (N, 6): x1, y1, x2, y2, track_id, cls. Past only: the state is
    the track histories and the last raw values. frame_wh is the size of the frame
    in which the boxes are given (to drop boxes clipped by the frame edge)."""

    def __init__(self, frame_wh: tuple[float, float] | None = None):
        self.frame_wh = frame_wh
        self.tracks: dict[int, deque] = {}
        self.raw_hist: deque = deque(maxlen=MEDIAN_K)
        self.score = 0.0
        self.explain: dict | None = None   # highest-risk pair at the last step

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
            d = self.tracks.setdefault(int(tid), deque(maxlen=HISTORY))
            # ByteTrack does not distinguish classes: a pedestrian's ID can pass to a car
            # that covered their box. A "pedestrian to car" velocity is a false jump,
            # so when the class family changes the track history starts over.
            if d and (int(d[-1][4]) in VEHICLE_CLS) != (int(cls) in VEHICLE_CLS):
                d.clear()
            d.append((t_sec, (x1 + x2) / 2.0, float(y2), diag, int(cls)))
        for tid in [k for k, d in self.tracks.items() if t_sec - d[-1][0] > STALE_SEC]:
            del self.tracks[tid]

    @staticmethod
    def _back(d, t, lag):
        """Last track point no later than t - lag."""
        for pt in reversed(d):
            if t - pt[0] >= lag:
                return pt
        return None

    def _state(self, d, t_sec):
        """(pos, vel, vel_prev|None, diag, cls) in px and px/s, or None if
        the track is not fresh or is shorter than VEL_BASELINE_SEC."""
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
        # Same class but very different box sizes: different depths, so no contact, however close
        # the boxes look in the image (C3902, 0:13.5: a near car heading for a car far down the road).
        ratio = np.maximum(diag[i], diag[j]) / np.maximum(np.minimum(diag[i], diag[j]), 1.0)
        keep &= ~((cls[i] == cls[j]) & (ratio > SAME_CLASS_MAX_SIZE_RATIO))
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return 0.0
        d_norm = ((diag[i] + diag[j]) / 2.0)[:, None]
        p = (pos[i] - pos[j]) / d_norm
        v = (vel[i] - vel[j]) / d_norm
        dist = np.linalg.norm(p, axis=1)
        v_c = -(p * v).sum(axis=1) / np.maximum(dist, 1e-6)        # > 0: closing in
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
        # closing speed half a second ago: if it is noticeably slower now, the pair is braking
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
    """Detector on every stride-th frame + RiskScorer.

    Detection runs in a background thread while the harness decodes the next frames
    (4K decoding is the main time cost, and it does not wait for the GPU). Detection
    of frame k is launched in step(k), and its result is collected strictly in the
    next step() that runs inference — so the result does not depend on thread
    speed (determinism), and the score at time t uses only frames before t
    (causality). The price is a one-step delay: stride frames, ~0.07 s.
    """
    _model = None  # weights are loaded once per process
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
        self._drain()                      # tail of the previous video, if any
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
        self._pending = None               # (future, t_sec) of the detection in flight
        self.n_failed = 0
        self.explain_log = None            # demo: list of (t, score, risk pair) for rendering
        self._base = None                  # (wall, idx): where pace measurement starts
        self._over = 0                     # consecutive checks with the projection past the deadline

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.idx += 1
        if self.idx % 50 == 0:
            self._replan(time.perf_counter())
        if self.disabled or self.idx % self.stride != 0:
            return self.last_score
        self._collect()
        self._pending = (self._pool.submit(self._detect, frame), t_sec)
        return self.last_score

    # ------------------------------------------------------------ detection
    def _collect(self) -> None:
        """Collects the detection launched at the previous step and updates the score."""
        if self._pending is None:
            return
        fut, t_det = self._pending
        self._pending = None
        try:
            dets = fut.result()
        except Exception as exc:  # one bad frame must not bring down the whole run
            self.n_failed += 1
            if self.n_failed == 1:
                print(f"[risk] detector failed at t={t_det:.1f}s: {exc!r} (further failures are not logged)")
            return
        self.last_score = self.scorer.feed(dets, t_det)
        if self.explain_log is not None:   # visualisation only (demo); does not affect the score
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
        """(N, 6) float32: x1, y1, x2, y2, track_id, cls in cropped-frame coordinates."""
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

    # ------------------------------------------------------------ budget
    def _replan(self, now: float) -> None:
        """Projects the end of the video from the AVERAGE pace (harness decode + our inference)
        since the end of warm-up. Falling behind — detector runs less often; deadline passed —
        detector is switched off and the score is 0 (the stale score would become one long alarm).

        Pace over the last 50 frames was too jumpy: one slow second of
        decoding gave a projection of "+19 s past the deadline" where the video actually finished
        with 300 s to spare (C3902), stride grew, and the risk curve depended on
        whatever else was loading the machine. Now: the average since the start, and only two
        consecutive checks past the deadline trigger stride += 1 (and the pace is measured anew)."""
        if self.deadline == float("inf") or self.n_frames <= 0:
            return
        if now > self.deadline:
            if not self.disabled:
                print(f"[risk] budget exhausted at t={self.idx / self.fps:.0f}s — detector switched off")
            self.disabled = True
            self._drain()
            self.last_score = 0.0
            return
        if self.idx < REPLAN_WARMUP_FRAMES:        # first frames — CUDA warm-up
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
            print(f"[risk] projection {projected - self.deadline:+.0f}s past the deadline -> stride={self.stride}")
