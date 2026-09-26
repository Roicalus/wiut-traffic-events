"""light_state.py — classifies the traffic-light colour in the 'light_roi' ROI
from zones.json on every N-th frame of the video.

In the submission the light is read inside the tracker pass (LightScanner,
pipeline.extract): there is no separate video decode. run_light_state() and the
CLI (main()) are for debugging only: a separate pass, JSON and the state
distribution, to check that the ROI has not missed the traffic light.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MIN_BRIGHT_PIXELS = 15  # fewer bright pixels of the right colour than this -> "unknown",
                          # we do not guess on noise in a small ROI


def classify_roi_hue(roi_bgr, min_pixels=MIN_BRIGHT_PIXELS) -> str:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    red_mask = (cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                | cv2.inRange(hsv, (170, 100, 100), (180, 255, 255)))
    yellow_mask = cv2.inRange(hsv, (18, 100, 100), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (45, 80, 80), (90, 255, 255))
    counts = {
        "red": int((red_mask > 0).sum()),
        "yellow": int((yellow_mask > 0).sum()),
        "green": int((green_mask > 0).sum()),
    }
    best = max(counts, key=counts.get)
    if counts[best] < min_pixels:
        return "unknown"
    return best


# Classification mode:
#   "position" — which of the three sections (top/middle/bottom) is lit. Independent
#                of the lighting hue (sunset, headlights, brake lights behind the box).
#                Requires light_roi to fit tightly around the VERTICAL housing of
#                the traffic light (red on top, green at the bottom).
#   "hue"      — the old way: count pixels of the right colour in the box.
#   "auto"     — position if the box is elongated vertically (h/w >= 1.5).
LIGHT_MODE = "auto"
# In daytime sun a lit lamp is dim: the 98th percentile of V for the lit section is
# 45-115 vs 0-40 for unlit ones (C3896, C3897); at sunset/dusk it is 255.
# So the DIFFERENCE from the neighbouring section decides, and the absolute threshold is only
# the "nothing is lit" cut-off. It used to be 150: daytime clips were "unknown" throughout.
# With these thresholds all 4 samples read the same cycle:
# red ~37 s -> green ~37 s -> yellow 3 s.
LAMP_MIN_V = 40          # brightness of a lit lamp (0-255), below this it is off
LAMP_MIN_S = 70          # a lit lamp is saturated, the grey housing/asphalt is not
LAMP_MARGIN = 35         # how much brighter the lit section is than the others
INNER_X, INNER_Y = 0.2, 0.04   # trim the box edges: background gets in there (sky, foliage)
OCCLUDE_FRAC = 0.15      # fraction of the light box under a vehicle box = occlusion
MIN_CONFIDENT_FRAC = 0.3  # fewer confident readings than this — the box is not on the light, do not trust it
OCCLUDER_BELOW_PX = 60   # bottom of the occluding vehicle's box is below the bottom of the box (it is closer)


def _lamp_scores(roi_bgr):
    h0, w0 = roi_bgr.shape[:2]
    dx, dy = int(w0 * INNER_X), int(h0 * INNER_Y)
    if w0 - 2 * dx >= 4 and h0 - 2 * dy >= 6:
        roi_bgr = roi_bgr[dy:h0 - dy, dx:w0 - dx]
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi_bgr, (3, 3), 0), cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    v[hsv[..., 1] < LAMP_MIN_S] = 0          # saturated (coloured) pixels only
    h = v.shape[0]
    bands = (v[: h // 3], v[h // 3: 2 * h // 3], v[2 * h // 3:])
    return [float(np.percentile(b, 98)) if b.size else 0.0 for b in bands]


def classify_roi_position(roi_bgr):
    scores = _lamp_scores(roi_bgr)
    order = np.argsort(scores)[::-1]
    best, second = scores[order[0]], scores[order[1]]
    if best < LAMP_MIN_V or best - second < LAMP_MARGIN:
        return "unknown"
    return ("red", "yellow", "green")[int(order[0])]


def classify_roi(roi_bgr, min_pixels=MIN_BRIGHT_PIXELS, mode=None) -> str:
    mode = mode or LIGHT_MODE
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"
    if mode == "auto":
        h, w = roi_bgr.shape[:2]
        mode = "position" if h >= 1.5 * w else "hue"
    if mode == "position":
        return classify_roi_position(roi_bgr)
    return classify_roi_hue(roi_bgr, min_pixels)


def bbox_from_zone(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


SKIP_YELLOW_CONFIRM_SEC = 3.0   # green -> red without yellow: only if it holds this long
HOLD_MAX_SEC = 45.0             # longer than a phase without a confident reading — colour is unknown


def smooth_states(raw, confirm_frames=2, skip_yellow_sec=SKIP_YELLOW_CONFIRM_SEC,
                  hold_max_sec=HOLD_MAX_SEC):
    """Sticky-hold smoothing: [(t, state)] -> [[t, state]].

    The last CONFIDENT (not "unknown") colour is held until a
    DIFFERENT confident colour appears confirm_frames times in a row. "unknown" (glare,
    ROI occlusion, dark frame) does not reset the state.

    The light cycle is green -> yellow (3 s) -> red. A direct jump
    green -> red is almost always an artefact: a vehicle covered the green
    section, and the dim unlit red lens reads as lit in daytime
    (C3896, 189.8-192.5 s — a false red_light). Such a transition is accepted
    only if red holds for skip_yellow_sec (yellow is sometimes
    missed at 10 Hz — then red is simply late by those seconds).

    With no confident readings for longer than hold_max_sec (longer than one phase: ROI
    covered, glare, camera moved) the colour is reset to "unknown", otherwise a single
    stale "red" would be held for minutes and produce false red_light events.
    """
    result = []
    last_known = "unknown"
    candidate, candidate_count, candidate_t0 = None, 0, None
    last_seen = None
    for t, state in raw:
        if state == "unknown":
            if last_seen is not None and t - last_seen > hold_max_sec:
                last_known = "unknown"
        elif state == last_known:
            last_seen = t
            candidate, candidate_count = None, 0
        else:
            if state == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count, candidate_t0 = state, 1, t
            skips_yellow = last_known == "green" and candidate == "red"
            if candidate_count >= confirm_frames and (
                    not skips_yellow or t - candidate_t0 >= skip_yellow_sec):
                last_known, last_seen = candidate, t
                candidate, candidate_count = None, 0
        result.append([round(t, 3), last_known])
    return result


class LightScanner:
    """Streaming traffic-light classifier: fed with frames from the tracker
    pass (track.run_tracker(on_frame=...)), so no separate decode of the
    4K video is needed — this saves a whole pass over the clip.

    The frame may be cropped at the top: pass full_height and the offset
    is computed automatically (as in ObstacleFireScanner).
    """

    def __init__(self, zones, full_height=None, min_pixels=MIN_BRIGHT_PIXELS):
        if "light_roi" not in zones:
            raise ValueError("zones has no 'light_roi'")
        self.box = bbox_from_zone(zones["light_roi"])
        self.n_occluded = 0
        self.full_height = full_height
        self.min_pixels = min_pixels
        self.raw = []
        self.failed = False

    def feed(self, frame, t_sec):
        y_off = (self.full_height - frame.shape[0]) if self.full_height else 0
        x1, y1, x2, y2 = self.box
        roi = frame[max(0, y1 - y_off):max(0, y2 - y_off), max(0, x1):max(0, x2)]
        state = classify_roi(roi, self.min_pixels) if roi.size else "unknown"
        self.raw.append((t_sec, state))

    def occluded(self, result, y_off) -> bool:
        """The ROI is covered by a vehicle standing CLOSER to the camera than the light:
        the box covers >= OCCLUDE_FRAC of the light box, and its bottom (ground point)
        is more than OCCLUDER_BELOW_PX below the bottom of the light box. Vehicles in the far
        lanes behind the light also cross the box in the image, but they sit
        higher in the frame and cover nothing."""
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return False
        x1, y1, x2, y2 = self.box
        area = max((x2 - x1) * (y2 - y1), 1)
        for bx1, by1, bx2, by2 in result.boxes.xyxy.cpu().numpy():
            by1, by2 = by1 + y_off, by2 + y_off
            inter = max(0.0, min(x2, bx2) - max(x1, bx1)) * max(0.0, min(y2, by2) - max(y1, by1))
            if inter / area >= OCCLUDE_FRAC and by2 > y2 + OCCLUDER_BELOW_PX:
                return True
        return False

    def on_tracker_frame(self, cropped_frame, t_sec, result=None):
        if self.failed:
            return
        try:
            y_off = (self.full_height - cropped_frame.shape[0]) if self.full_height else 0
            if self.occluded(result, y_off):
                self.n_occluded += 1
                self.raw.append((t_sec, "unknown"))
                return
            self.feed(cropped_frame, t_sec)
        except Exception as exc:
            self.failed = True
            print(f"[light_state] failed ({exc}); red_light/stop_line skipped")

    def samples(self, confirm_frames=2):
        """Smoothed state series, or None if the light cannot be trusted:
        the scanner failed or confident readings are below MIN_CONFIDENT_FRAC (on
        the samples it is 91-100%; low means the box is not on the light: alignment
        went wrong or the light was replaced). Then red_light/stop_line are not
        emitted for the video, and congestion is computed from duration — better than
        events based on a garbage "colour"."""
        if self.failed or not self.raw:
            return None
        seen = [s for _, s in self.raw if s != "unknown"]
        frac = len(seen) / max(len(self.raw) - self.n_occluded, 1)
        if frac < MIN_CONFIDENT_FRAC:
            print(f"[light_state] traffic light read confidently in only {frac:.0%} of frames — "
                  f"not used (no red_light/stop_line for this video)")
            return None
        return smooth_states(self.raw, confirm_frames)


def run_light_state(video_path, zones, stride=3, min_pixels=MIN_BRIGHT_PIXELS,
                     confirm_frames=2):
    """Separate pass over the video (CLI/debugging only; in pipeline.extract
    the light is read inside the tracker pass via LightScanner).
    Returns a list of [t_sec, "red"|"yellow"|"green"|"unknown"]."""
    scanner = LightScanner(zones, min_pixels=min_pixels)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = -1
    while cap.grab():
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        scanner.feed(frame, frame_idx / fps)
    cap.release()
    return scanner.samples(confirm_frames)


# ---------------------------------------------------------------- CLI (dev only)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--confirm-frames", type=int, default=2,
                     help="number of consecutive confident (not unknown) samples of a new colour "
                          "needed to switch state (1 = on the very first one)")
    ap.add_argument("--min-pixels", type=int, default=MIN_BRIGHT_PIXELS)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path("src/light") / (Path(args.video).stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zones = json.loads(Path(args.zones).read_text())
    smoothed = run_light_state(args.video, zones, stride=args.stride,
                                confirm_frames=args.confirm_frames, min_pixels=args.min_pixels)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video": Path(args.video).name, "fps": fps, "samples": smoothed}, f, indent=1)

    counts = {}
    for _, s in smoothed:
        counts[s] = counts.get(s, 0) + 1
    print(f"{len(smoothed)} samples -> {out_path}")
    print("State distribution:", counts)
    if counts.get("unknown", 0) == len(smoothed):
        print("WARNING: the whole clip is 'unknown' — the light was never confidently "
              "recognised, check the ROI (light_roi) and/or --min-pixels")
    elif counts.get("red", 0) == 0:
        print("WARNING: 'red' never occurred — check the ROI (light_roi) and/or --min-pixels")


if __name__ == "__main__":
    main()
