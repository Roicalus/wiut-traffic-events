"""
obstacle_fire.py — road_obstacle and fire_smoke detectors in ONE pass.

Both classes are pure CV without YOLO (background subtraction for obstacle,
an HSV heuristic for fire_smoke). They used to be two separate passes over the
4K video, each at full resolution. Now:

  * a frame is decoded once and immediately downscaled to SCAN_WIDTH in width;
    MOG2, morphology and HSV run on the downscaled frame (~16x fewer
    pixels than 4K);
  * the "blob already covered by a tracked object" filter runs AFTER the pass
    and is vectorized (previously a loop over all tracker records for every blob);
  * ObstacleFireScanner can be attached to the tracker pass via on_frame
    (see pipeline.extract) — then there is no separate video decoding at all;
  * progress is printed to the console.

All thresholds below are in pixels of the original 4K frame (3840x2160) and
are rescaled to the downscaled frame automatically.

Both detectors are a first version, NOT calibrated against the labels. If they
mostly miss on evaluate.py, the classes are not included in solution.CLASSES.
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


# ---------------------------------------------------------------- common
SCAN_WIDTH = 960             # width of the frame both detectors run on
SAMPLE_SEC = 0.15            # process at most once per this many seconds of video
PROGRESS_EVERY_SEC = 60.0    # print progress once per this many seconds of video

# ---------------------------------------------------------------- road_obstacle
MIN_OBSTACLE_AREA = 900      # px^2 at the original resolution
MAX_OBSTACLE_AREA_FRAC = 0.08  # fraction of the roadway zone area ("the whole frame became foreground")
STABLE_SEC = 4.0             # a blob must stay in place for this many seconds
MIN_EVENT_SEC = 4.0          # shorter ones are dropped as noise
MAX_CENTER_DRIFT_PX = 60.0   # blob centre drift (original px)
MAX_GAP_SEC = 2.0            # gap between samples of the same blob
MOG_LEARNING_RATE = 0.0005    # rate at which a static object is "absorbed" into the background (sample = ~0.2 s)
COVERED_RADIUS_PX = 80.0     # a blob closer than this to a tracked object is not an obstacle
COVERED_WINDOW_SEC = 1.0

# ---------------------------------------------------------------- fire_smoke
FIRE_MIN_PIXELS = 250        # "fire" pixels (scaled to the 4K frame)
SMOKE_MIN_PIXELS = 4000      # grey smoke pixels (scaled to the 4K frame)
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
    """Streaming scanner: feed frames in order via feed(), then at the end
    call finish(records) -> list of road_obstacle + fire_smoke events.

    frame may be cropped at the top (as in track.run_tracker): pass
    full_height — the height of the FULL frame — and the scanner computes the offset itself.
    Zones are given in full-frame coordinates.
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
        self._mask = None            # roadway mask (SCAN_WIDTH scale), None -> obstacle disabled
        self._blob_samples = []      # [(t, [(cx, cy) in original px, ...]), ...]
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

    # ------------------------------------------------------------ run
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
            fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]  # MOG2 shadows = 127
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
        """Callback for track.run_tracker(on_frame=...). A scanner error must not
        crash the tracker pass: on the first exception the scanner
        is disabled and road_obstacle/fire_smoke are skipped for the video."""
        try:
            self.feed(cropped_frame, t_sec)
        except Exception as exc:
            self.failed = True
            print(f"[obstacle_fire] scanner crashed ({exc}); road_obstacle/fire_smoke skipped")

    # ------------------------------------------------------------ results
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
    """Merges adjacent events of the same class with a gap <= max_gap."""
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


# ---------------------------------------------------------------- standalone pass
def scan_video(video_path, zones, records=None, progress: bool = True) -> list[list]:
    """One dedicated pass over the video (when the scanner is not attached to the tracker)."""
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
                    help="src/tracks/*.json from track.py (for the 'already covered by a "
                         "tracked object' filter; without it the filter is off)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    zones = load_zones(args.zones)
    records = json.loads(Path(args.tracks).read_text())["tracks"] if args.tracks else []

    t0 = time.perf_counter()
    events = sorted(scan_video(args.video, zones, records))
    elapsed = time.perf_counter() - t0

    n_obs = sum(1 for e in events if e[2] == "road_obstacle")
    print(f"road_obstacle: {n_obs}, fire_smoke: {len(events) - n_obs}, time {elapsed:.0f} s")
    for s, e, lbl in events:
        print(f"  [{s:7.2f} - {e:7.2f}] {lbl}")
    if args.out:
        Path(args.out).write_text(json.dumps(events, indent=1))
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
