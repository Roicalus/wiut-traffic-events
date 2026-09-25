"""check_alignment.py — насколько камера сдвинута в каждом видео и легли ли зоны.

    python tools/check_alignment.py --videos samples

Для каждого видео печатает найденный сдвиг и пишет debug/align_<видео>.jpg:
слева — кадр с совмещёнными зонами, справа — крупно светофор (light_roi)
ДО (красная рамка) и ПОСЛЕ (зелёная) совмещения. Зелёная рамка должна
плотно накрывать корпус светофора на КАЖДОМ видео.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import align  # noqa: E402
from src.rules import load_zones  # noqa: E402


def bbox(poly, pad=0):
    xs, ys = poly[:, 0], poly[:, 1]
    return int(xs.min()) - pad, int(ys.min()) - pad, int(xs.max()) + pad, int(ys.max()) + pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "debug"))
    args = ap.parse_args()
    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zones = load_zones(ROOT / "zones.json")

    for path in videos:
        frame = align.first_frame(path)
        if frame is None:
            print(f"[{path.name}] не читается")
            continue
        M, rep = align.estimate_video(path)      # то же, что в pipeline
        z_new = align.transform_zones(zones, M)
        print(f"[{path.name}] {rep}")

        vis = frame.copy()
        for name, poly in z_new.items():
            closed = len(poly) >= 3
            cv2.polylines(vis, [poly.astype(np.int32)], closed, (0, 255, 0), 3)
            cv2.putText(vis, name, tuple(int(v) for v in poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 255, 0), 3)
        h, w = vis.shape[:2]
        s = 1600 / w
        left = cv2.resize(vis, (1600, int(h * s)))

        if "light_roi" in zones:
            x1, y1, x2, y2 = bbox(np.vstack([zones["light_roi"], z_new["light_roi"]]), pad=120)
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2].copy()
            for poly, col in ((zones["light_roi"], (0, 0, 255)), (z_new["light_roi"], (0, 255, 0))):
                cv2.polylines(crop, [(poly - (x1, y1)).astype(np.int32)], True, col, 2)
            cz = cv2.resize(crop, (int(crop.shape[1] * left.shape[0] / crop.shape[0]), left.shape[0]))
            left = np.hstack([left, cz])
        txt = f"{path.name}: {rep.get('status')} dx={rep.get('dx')} dy={rep.get('dy')} inl={rep.get('inliers')}"
        cv2.putText(left, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out = out_dir / f"align_{path.stem}.jpg"
        cv2.imwrite(str(out), left)
        print(f"    -> {out}")


if __name__ == "__main__":
    main()
