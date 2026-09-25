"""light_check.py — проверка распознавания светофора глазами.

    python tools/light_check.py --videos samples --every 4

Для каждого видео: кроп light_roi (после совмещения зон) каждые --every
секунд, подписи двух методов (P = position, H = hue) -> одна картинка-сетка
debug/light_<видео>.jpg. Пролистайте: подпись P должна совпадать с тем,
какая секция реально горит. Внизу печатается, как часто методы расходятся.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import align  # noqa: E402
from src.light_state import bbox_from_zone, classify_roi  # noqa: E402
from src.rules import load_zones  # noqa: E402

COL = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (0, 255, 0), "unknown": (160, 160, 160)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--every", type=float, default=4.0, help="шаг по времени, с")
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--out-dir", default=str(ROOT / "debug"))
    args = ap.parse_args()
    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    zones = load_zones(ROOT / "zones.json")

    for path in videos:
        z, rep = align.aligned_zones(path, zones)
        x1, y1, x2, y2 = bbox_from_zone(z["light_roi"])
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles, stats = [], Counter()
        for f in range(0, n, max(1, int(args.every * fps))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            p, h = classify_roi(roi, mode="position"), classify_roi(roi, mode="hue")
            stats[(p, h)] += 1
            tile = cv2.resize(roi, (90, int(90 * roi.shape[0] / max(roi.shape[1], 1))))
            tile = cv2.copyMakeBorder(tile, 0, 46, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            th = tile.shape[0]
            cv2.putText(tile, f"{f / fps:.0f}s", (3, th - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(tile, f"P:{p[:3]}", (3, th - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL[p], 1)
            cv2.putText(tile, f"H:{h[:3]}", (3, th - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL[h], 1)
            tiles.append(tile)
        cap.release()
        if not tiles:
            continue
        th = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % args.cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
        out = Path(args.out_dir) / f"light_{path.stem}.jpg"
        cv2.imwrite(str(out), np.vstack(rows))
        disagree = sum(v for (p, h), v in stats.items() if p != h)
        print(f"[{path.name}] совмещение: {rep.get('status')} dx={rep.get('dx')} dy={rep.get('dy')}; "
              f"методы расходятся в {disagree}/{sum(stats.values())} кадрах -> {out}")
        print("    ", dict(sorted(Counter(p for p, _ in stats.elements()).items())), "(position)")


if __name__ == "__main__":
    main()
