"""
define_zones.py — define the camera's zone polygons by clicking on a single frame
(the queue zone in front of the traffic light, crossing zones, the roadway, the traffic-light
ROI, etc.). The result is zones.json, reused for ALL videos from this camera (the samples and
the hidden test share the same view).

Usage:
    python define_zones.py --video samples/C3896.MP4 --frame 150 --out zones.json

Controls:
    LMB     — add a point to the current polygon
    RMB     — close the current polygon (needs >=3 points; not needed for line 7
              (solid_line) — it is not closed, just 2 points)
    1..8    — assign a name from the list below to the polygon just closed
              (after that you can start the next polygon right away);
              7 (solid_line) is a special case: it is a LINE (exactly 2 points),
              not a polygon, see src/rules.py detect_solid_line_crossing
    0       — assign a custom name (the only case where it asks in the
              console — use only for non-standard zones)
    U / Ctrl+Z — undo: the last point of the current polygon, or the last
              saved zone if the current polygon is empty (its points come back)
    Q       — finish and save everything to --out

Click coordinates are stored in the ORIGINAL video resolution, even if the
image is shown scaled down on screen (see --max-width/--max-height) —
zones.json stays valid for track.py/rules.py, which work on
full-size frames.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRESET_NAMES = {
    ord('1'): "queue_zone",     # queue across all lanes in front of the crossing — for congestion
    ord('2'): "roadway",        # roadway outside the crossings — for jaywalking
    ord('3'): "stop_line",      # stop line in front of the far crossing
    ord('4'): "crossing_far",   # crossing over the main road
    ord('5'): "crossing_near",  # diagonal crossing across the square
    ord('6'): "light_roi",      # box around the traffic-light signal — for colour detection
    ord('7'): "solid_line",     # LINE (exactly 2 points!) — for solid_line_crossing
    ord('8'): "illegal_turn_exit",  # exit that cannot be reached from the main road (see rules.detect_illegal_turn)
}

# LINE zones: not a polygon but an open segment — 2 points are enough, no need to
# close with a right-click (see LINE_MIN_POINTS below and src/rules.py,
# detect_solid_line_crossing — it also explains why this is a line
# and not a polygon: cv2.pointPolygonTest cannot tell "which side of a line").
LINE_PRESETS = {"solid_line"}


def _min_points_for(name: str) -> int:
    if name in LINE_PRESETS or name.startswith("solid_line_"):
        return 2
    return 3


def fit_scale(w, h, max_w, max_h):
    return min(max_w / w, max_h / h, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=150,
                     help="frame number to annotate (a frame with traffic is better)")
    ap.add_argument("--out", default="zones.json")
    ap.add_argument("--load", action="store_true",
                     help="load the existing --out and add/replace zones (a zone with the same "
                          "name is overwritten). Draw on the SAME video/frame as in zones_ref.json")
    ap.add_argument("--max-width", type=int, default=1600,
                     help="max. on-screen window width (the image is scaled to fit it)")
    ap.add_argument("--max-height", type=int, default=900,
                     help="max. on-screen window height")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= n_frames:
        cap.release()
        raise SystemExit(f"Frame {args.frame} is beyond the end of the video (total frames: {n_frames})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {args.frame} from {args.video}")

    h, w = frame.shape[:2]
    scale = fit_scale(w, h, args.max_width, args.max_height)
    print(f"Frame {w}x{h}, display scale: {scale:.3f} "
          f"(coordinates in zones.json will be in the original resolution {w}x{h})")
    disp_w, disp_h = int(w * scale), int(h * scale)

    zones = {}          # name -> polygon in ORIGINAL video coordinates
    if args.load and Path(args.out).exists():
        zones = json.loads(Path(args.out).read_text())
        print(f"Loaded {len(zones)} zones from {args.out}: {list(zones)}")
        ref_meta = Path(__file__).resolve().parent.parent / "zones_ref.json"
        if ref_meta.exists():
            m = json.loads(ref_meta.read_text())
            if (m.get("video"), m.get("frame")) != (Path(args.video).name, args.frame):
                print(f"WARNING: the zones' reference frame is {m.get('video')} frame {m.get('frame')}, but you "
                      f"are drawing on {Path(args.video).name} frame {args.frame}. The camera is shifted "
                      f"between videos — draw on the reference, or the new zones will not match the old ones.")
    current = []         # current unclosed polygon, also in original coordinates
    history = []         # action stack for undo: ('point',) | ('zone', name, polygon)
    status = "LMB: click polygon points, RMB: close"

    def undo():
        nonlocal current, status
        if not history:
            status = "Nothing to undo"
            return
        action = history.pop()
        if action[0] == "point":
            if current:
                current.pop()
            status = "Last point undone"
        elif action[0] == "zone":
            _, name, poly = action
            zones.pop(name, None)
            current = poly  # the points go back into work — can be redrawn/renamed
            status = f"Zone '{name}' undone — points are back in work, assign the name again"

    def redraw():
        vis = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        for name, poly in zones.items():
            pts = (np.array(poly, dtype=np.float32) * scale).astype(int)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.putText(vis, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        if current:
            pts = (np.array(current, dtype=np.float32) * scale).astype(int)
            for p in pts:
                cv2.circle(vis, tuple(p), 4, (0, 0, 255), -1)
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, (0, 0, 255), 2)
        legend = " | ".join(f"{k}:{v}" for k, v in
                             {"1": "queue_zone", "2": "roadway", "3": "stop_line",
                              "4": "crossing_far", "5": "crossing_near", "6": "light_roi",
                              "7": "solid_line(2pt)", "8": "illegal_turn_exit"}.items())
        cv2.putText(vis, status, (10, disp_h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 0), 1)
        cv2.putText(vis, legend + "  |  U/Ctrl+Z: undo", (10, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("define_zones", vis)

    def on_mouse(event, x, y, flags, param):
        nonlocal status
        if event == cv2.EVENT_LBUTTONDOWN:
            # screen coordinates -> original video resolution
            current.append([x / scale, y / scale])
            history.append(("point",))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current) < 3:
                status = "At least 3 points are needed to close a polygon (not for line 7 — press 7 right after 2 points)"
            else:
                status = "Polygon closed — press 1-8 (preset name) or 0 (custom)"
            redraw()

    cv2.namedWindow("define_zones", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("define_zones", disp_w, disp_h)
    cv2.setMouseCallback("define_zones", on_mouse)
    redraw()

    while True:
        redraw()
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('u'), 26):  # 'u' or Ctrl+Z
            undo()
        elif key in PRESET_NAMES and len(current) >= _min_points_for(PRESET_NAMES[key]):
            name = PRESET_NAMES[key]
            zones[name] = current
            history.append(("zone", name, current))
            current = []
            status = f"Saved: {name} — draw the next polygon"
        elif key == ord('0') and len(current) >= 2:
            cv2.destroyWindow("define_zones")  # so the console input() is definitely in focus
            name = input("Custom zone name: ").strip()
            cv2.namedWindow("define_zones", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("define_zones", disp_w, disp_h)
            cv2.setMouseCallback("define_zones", on_mouse)
            if name:
                zones[name] = current
                history.append(("zone", name, current))
                status = f"Saved: {name}"
            current = []
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(zones, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(zones)} zone(s) to {args.out}: {list(zones.keys())}")
    try:
        from src import align
        align.save_reference(frame, Path(args.video).name, args.frame)
        print(f"Reference frame for zone alignment: {align.REF_IMAGE} (commit it together with zones.json)")
    except Exception as exc:
        print(f"Reference frame not saved: {exc}")


if __name__ == "__main__":
    main()