"""
obstacle_fire.py — детекторы road_obstacle и fire_smoke за ОДИН проход.

Оба класса — чистый CV без YOLO (background subtraction для obstacle,
HSV-эвристика для fire_smoke). Раньше это были два отдельных прохода по
4K-видео, каждый на полном разрешении. Теперь:

  * кадр декодируется один раз и сразу уменьшается до SCAN_WIDTH по ширине;
    MOG2, морфология и HSV работают на уменьшенном кадре (в ~16 раз меньше
    пикселей, чем на 4K);
  * фильтр "блоб уже покрыт трекнутым объектом" выполняется ПОСЛЕ прохода
    и векторизован (раньше — цикл по всем записям трекера на каждый блоб);
  * ObstacleFireScanner можно подключить к проходу трекера через on_frame
    (см. pipeline.extract) — тогда отдельного декодирования видео нет совсем;
  * прогресс печатается в консоль.

Все пороги ниже заданы в пикселях исходного 4K-кадра (3840x2160) и
пересчитываются под уменьшенный кадр автоматически.

Оба детектора — первая версия, НЕ откалибрована против разметки. Если на
evaluate.py они в основном мажут — классы не включены в solution.CLASSES.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from src.rules import load_zones, _zones_by_prefix
except ImportError:
    from rules import load_zones, _zones_by_prefix


# ---------------------------------------------------------------- общие
SCAN_WIDTH = 960             # ширина кадра, на котором работают оба детектора
SAMPLE_SEC = 0.15            # обрабатывать не чаще, чем раз в столько секунд видео
PROGRESS_EVERY_SEC = 60.0    # печатать прогресс раз в столько секунд видео

# ---------------------------------------------------------------- road_obstacle
MIN_OBSTACLE_AREA = 900      # px^2 в исходном разрешении
MAX_OBSTACLE_AREA_FRAC = 0.08  # доля площади roadway-зоны ("весь кадр стал foreground")
STABLE_SEC = 4.0             # блоб должен держаться на месте столько секунд
MIN_EVENT_SEC = 4.0          # короче — дропаем как шум
MAX_CENTER_DRIFT_PX = 60.0   # дрейф центра блоба (исходные px)
MAX_GAP_SEC = 2.0            # разрыв между сэмплами того же блоба
MOG_LEARNING_RATE = 0.0005    # скорость "впитывания" неподвижного предмета в фон (сэмпл = ~0.2 с)
COVERED_RADIUS_PX = 80.0     # блоб ближе этого к трекнутому объекту — не obstacle
COVERED_WINDOW_SEC = 1.0

# ---------------------------------------------------------------- fire_smoke
FIRE_MIN_PIXELS = 250        # "огненных" пикселей (в пересчёте на 4K-кадр)
SMOKE_MIN_PIXELS = 4000      # серо-дымных пикселей (в пересчёте на 4K-кадр)
CONFIRM_SAMPLES = 3
MAX_GAP_SEC_FIRE = 3.0
MIN_EVENT_SEC_FIRE = 2.0


def _odd(x: float, lo: int = 3) -> int:
    k = max(lo, int(round(x)))
    return k if k % 2 == 1 else k + 1


def _fire_pixels(hsv):
    m1 = cv2.inRange(hsv, (0, 120, 180), (25, 255, 255))
    m2 = cv2.inRange(hsv, (160, 120, 180), (180, 255, 255))
    return cv2.bitwise_or(m1, m2)


def _smoke_pixels(hsv):
    _, s, v = cv2.split(hsv)
    return ((s < 60) & (v > 90) & (v < 240)).astype(np.uint8) * 255


class ObstacleFireScanner:
    """Потоковый сканер: кормите кадрами по порядку через feed(), в конце
    вызовите finish(records) -> список событий road_obstacle + fire_smoke.

    frame может быть обрезан сверху (как в track.run_tracker): передайте
    full_height — высоту ПОЛНОГО кадра, и сканер сам вычислит смещение.
    Зоны заданы в координатах полного кадра.
    """

    def __init__(self, zones, fps: float, full_height: int | None = None, progress: bool = True):
        self.zones = zones
        self.fps = fps
        self.full_height = full_height
        self.progress = progress
        self.failed = False

        self._ready = False
        self._last_t = -1e9
        self._next_progress = PROGRESS_EVERY_SEC
        self._bg = None
        self._mask = None            # маска проезжей части (масштаб SCAN_WIDTH), None -> obstacle выключен
        self._blob_samples = []      # [(t, [(cx, cy) в исходных px, ...]), ...]
        self._fire_raw = []          # [(t, is_candidate), ...]

    # ------------------------------------------------------------ setup
    def _setup(self, frame):
        h, w = frame.shape[:2]
        self._scale = min(1.0, SCAN_WIDTH / w)
        self._dw, self._dh = int(round(w * self._scale)), int(round(h * self._scale))
        self._y_off = (self.full_height - h) if self.full_height else 0
        area_scale = self._scale ** 2
        self._min_area = MIN_OBSTACLE_AREA * area_scale
        self._fire_min = FIRE_MIN_PIXELS * area_scale
        self._smoke_min = SMOKE_MIN_PIXELS * area_scale
        self._k_open = np.ones((_odd(5 * self._scale),) * 2, np.uint8)
        self._k_close = np.ones((_odd(15 * self._scale),) * 2, np.uint8)

        names = _zones_by_prefix(self.zones, ("roadway", "crossroad"))
        if names:
            mask = np.zeros((self._dh, self._dw), dtype=np.uint8)
            for name in names:
                pts = (np.asarray(self.zones[name], dtype=np.float32) - (0.0, self._y_off)) * self._scale
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            self._mask = mask
            self._max_area = float((mask > 0).sum()) * MAX_OBSTACLE_AREA_FRAC
            self._bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)
        self._ready = True

    # ------------------------------------------------------------ прогон
    def feed(self, frame, t_sec: float) -> None:
        if self.failed or t_sec - self._last_t < SAMPLE_SEC:
            return
        self._last_t = t_sec
        if not self._ready:
            self._setup(frame)
        if self.progress and t_sec >= self._next_progress:
            print(f"[obstacle_fire] {t_sec:.0f}s", flush=True)
            self._next_progress += PROGRESS_EVERY_SEC

        small = frame if self._scale == 1.0 else cv2.resize(
            frame, (self._dw, self._dh), interpolation=cv2.INTER_AREA)

        # --- road_obstacle
        if self._bg is not None:
            fg = self._bg.apply(small, learningRate=MOG_LEARNING_RATE)
            fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]  # тени MOG2 = 127
            fg = cv2.bitwise_and(fg, self._mask)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k_open)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._k_close)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < self._min_area or area > self._max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                cx = (x + bw / 2.0) / self._scale
                cy = (y + bh / 2.0) / self._scale + self._y_off
                blobs.append((cx, cy))
            self._blob_samples.append((t_sec, blobs))

        # --- fire_smoke
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        fire_count = int(np.count_nonzero(_fire_pixels(hsv)))
        is_candidate = fire_count >= self._fire_min
        if is_candidate:
            smoke_count = int(np.count_nonzero(_smoke_pixels(hsv)))
            is_candidate = fire_count >= self._fire_min * 2 or smoke_count >= self._smoke_min
        self._fire_raw.append((t_sec, is_candidate))

    def on_tracker_frame(self, cropped_frame, t_sec, _result) -> None:
        """Колбэк для track.run_tracker(on_frame=...). Ошибка сканера не
        должна ронять проход трекера: при первом же исключении сканер
        отключается, road_obstacle/fire_smoke для видео пропускаются."""
        try:
            self.feed(cropped_frame, t_sec)
        except Exception as exc:
            self.failed = True
            print(f"[obstacle_fire] сканер упал ({exc}); road_obstacle/fire_smoke пропущены")

    # ------------------------------------------------------------ итоги
    def finish(self, records=None) -> list[list]:
        if self.failed:
            return []
        return self._obstacle_events(records or []) + self._fire_events()

    def _obstacle_events(self, records) -> list[list]:
        if not self._blob_samples:
            return []
        if records:
            rt = np.fromiter((r["t_sec"] for r in records), dtype=np.float64, count=len(records))
            order = np.argsort(rt)
            rt = rt[order]
            rcx = np.fromiter(((r["x1"] + r["x2"]) / 2.0 for r in records), np.float64, len(records))[order]
            rcy = np.fromiter(((r["y1"] + r["y2"]) / 2.0 for r in records), np.float64, len(records))[order]

        def covered(cx, cy, t):
            if not records:
                return False
            lo = np.searchsorted(rt, t - COVERED_WINDOW_SEC, side="left")
            hi = np.searchsorted(rt, t + COVERED_WINDOW_SEC, side="right")
            if lo >= hi:
                return False
            d2 = (rcx[lo:hi] - cx) ** 2 + (rcy[lo:hi] - cy) ** 2
            return bool((d2 <= COVERED_RADIUS_PX ** 2).any())

        active, finished = [], []
        for t_sec, raw_blobs in self._blob_samples:
            blobs = [b for b in raw_blobs if not covered(b[0], b[1], t_sec)]
            matched = set()
            for cx, cy in blobs:
                best_i, best_d = None, None
                for i, cand in enumerate(active):
                    if i in matched:
                        continue
                    d = math.hypot(cx - cand["cx"], cy - cand["cy"])
                    if d <= MAX_CENTER_DRIFT_PX and (best_d is None or d < best_d):
                        best_i, best_d = i, d
                if best_i is not None:
                    cand = active[best_i]
                    cand["cx"], cand["cy"], cand["last_t"] = cx, cy, t_sec
                    matched.add(best_i)
                else:
                    active.append({"cx": cx, "cy": cy, "first_t": t_sec, "last_t": t_sec})

            still = []
            for cand in active:
                if t_sec - cand["last_t"] > MAX_GAP_SEC:
                    if cand["last_t"] - cand["first_t"] >= STABLE_SEC:
                        finished.append([cand["first_t"] + STABLE_SEC, cand["last_t"]])
                else:
                    still.append(cand)
            active = still

        for cand in active:
            if cand["last_t"] - cand["first_t"] >= STABLE_SEC:
                finished.append([cand["first_t"] + STABLE_SEC, cand["last_t"]])

        events = [[round(s, 2), round(e, 2), "road_obstacle"] for s, e in finished
                  if e - s >= MIN_EVENT_SEC]
        return _merge_intervals(events)

    def _fire_events(self) -> list[list]:
        events, run_start, last_hit, confirm = [], None, None, 0
        for t_sec, is_candidate in self._fire_raw:
            if is_candidate:
                confirm += 1
                if run_start is None and confirm >= CONFIRM_SAMPLES:
                    run_start = t_sec
                last_hit = t_sec
            else:
                confirm = 0
                if run_start is not None and t_sec - last_hit > MAX_GAP_SEC_FIRE:
                    events.append([run_start, last_hit, "fire_smoke"])
                    run_start, last_hit = None, None
        if run_start is not None:
            events.append([run_start, last_hit, "fire_smoke"])
        events = [[round(s, 2), round(e, 2), lbl] for s, e, lbl in events
                  if e - s >= MIN_EVENT_SEC_FIRE]
        return _merge_intervals(events)


def _merge_intervals(events, max_gap=1.0):
    """Сливает соседние события одного класса с разрывом <= max_gap."""
    if not events:
        return []
    events = sorted(events, key=lambda e: (e[2], e[0]))
    merged = []
    cur_s, cur_e, cur_lbl = events[0]
    for s, e, lbl in events[1:]:
        if lbl == cur_lbl and s <= cur_e + max_gap:
            cur_e = max(cur_e, e)
        else:
            merged.append([cur_s, cur_e, cur_lbl])
            cur_s, cur_e, cur_lbl = s, e, lbl
    merged.append([cur_s, cur_e, cur_lbl])
    return merged


# ---------------------------------------------------------------- автономный проход
def scan_video(video_path, zones, records=None, progress: bool = True) -> list[list]:
    """Один собственный проход по видео (если сканер не подключён к трекеру)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, math.ceil(SAMPLE_SEC * fps))
    scanner = ObstacleFireScanner(zones, fps, progress=progress)
    idx = -1
    while cap.grab():
        idx += 1
        if idx % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        scanner.feed(frame, idx / fps)
    cap.release()
    return scanner.finish(records)


# ---------------------------------------------------------------- CLI (dev only)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--tracks", default=None,
                    help="src/tracks/*.json из track.py (для фильтра 'уже покрыто "
                         "трекнутым объектом'; без него фильтр выключен)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    zones = load_zones(args.zones)
    records = json.loads(Path(args.tracks).read_text())["tracks"] if args.tracks else []

    t0 = time.perf_counter()
    events = sorted(scan_video(args.video, zones, records))
    elapsed = time.perf_counter() - t0

    n_obs = sum(1 for e in events if e[2] == "road_obstacle")
    print(f"road_obstacle: {n_obs}, fire_smoke: {len(events) - n_obs}, время {elapsed:.0f} с")
    for s, e, lbl in events:
        print(f"  [{s:7.2f} - {e:7.2f}] {lbl}")
    if args.out:
        Path(args.out).write_text(json.dumps(events, indent=1))
        print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
