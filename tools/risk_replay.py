"""risk_replay.py — калибровка Part B без повторного инференса.

1) dump: один проход по видео так же, как харнесс (каждый кадр декодируется,
   детектор RiskEstimator._detect — на каждом BASE_STRIDE-м), детекции
   кэшируются в cache/risk/<видео>.pkl.gz. Заодно печатает, сколько времени
   ушло на декодирование и на детектор.
2) score: RiskScorer (тот же код, что в сабмите) по кэшу — за секунды.
   Печатает долю кадров с риском >= 0.5, алармы по правилам evaluate.py
   (прогоны >= 0.5, склейка при паузе < 2 с) и самые длинные из них.

    python tools/risk_replay.py dump  --videos samples
    python tools/risk_replay.py score --videos samples
    python tools/risk_replay.py score --videos samples --set D_COLL=0.4 --set TTC_SAFE=4
    python tools/risk_replay.py score --videos samples --gt my_labels.json   # + Score B
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import solution  # noqa: E402,F401  (сиды, YOLO_OFFLINE)
from src import risk  # noqa: E402

CACHE = ROOT / "cache" / "risk"
THETA, MERGE_GAP = 0.5, 2.0   # как в evaluate.py


def list_videos(src: Path) -> list[Path]:
    return [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")


def dump(video: Path) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    est = risk.RiskEstimator()
    est.reset({"video_id": video.name, "fps": fps, "n_frames": n,
               "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
    frames, t_dec, t_det, idx = [], 0.0, 0.0, -1
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t_dec += time.perf_counter() - t0
        if not ok:
            break
        idx += 1
        if idx % risk.BASE_STRIDE:
            continue
        t0 = time.perf_counter()
        dets = est._detect(frame)
        t_det += time.perf_counter() - t0
        frames.append((idx, idx / fps, dets))
    cap.release()
    dur = n / fps
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE / f"{video.stem}.pkl.gz", "wb") as f:
        pickle.dump({"video": video.name, "fps": fps, "n_frames": idx + 1,
                     "width": est.meta["width"], "height": est.meta["height"],
                     "stride": risk.BASE_STRIDE, "frames": frames}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{video.name}] {dur:.0f}s видео: декод {t_dec:.0f}s ({t_dec / dur:.2f}x), "
          f"детектор {t_det:.0f}s ({t_det / dur:.2f}x), итого {(t_dec + t_det) / dur:.2f}x", flush=True)


def replay(cached: dict) -> list[list[float]]:
    """Кривая [t_sec, score] на каждый кадр — как её записал бы харнесс.
    Как в RiskEstimator: детекции кадра с инференсом попадают в скор на
    СЛЕДУЮЩЕМ кадре с инференсом (детекция идёт в фоне)."""
    w, h = cached["width"], cached["height"]
    scorer = risk.RiskScorer((w, h - int(h * risk.CROP_TOP_FRAC)))
    by_idx = {i: (t, d) for i, t, d in cached["frames"]}
    fps, out, score, pending = cached["fps"], [], 0.0, None
    for idx in range(cached["n_frames"]):
        if idx in by_idx:
            if pending is not None:
                score = scorer.feed(pending[1], pending[0])
            pending = by_idx[idx]
        out.append([round(idx / fps, 3), score])
    return out


def alarms(curve: list[list[float]]) -> list[tuple[float, float, float]]:
    runs, cur = [], None
    for t, s in curve:
        if s >= THETA:
            cur = [t, t, s] if cur is None else [cur[0], t, max(cur[2], s)]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < MERGE_GAP:
            merged[-1] = [merged[-1][0], r[1], max(merged[-1][2], r[2])]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def apply_overrides(pairs: list[str]) -> None:
    for kv in pairs:
        k, v = kv.split("=", 1)
        if not hasattr(risk, k):
            raise SystemExit(f"в src/risk.py нет параметра {k}")
        setattr(risk, k, type(getattr(risk, k))(float(v)) if not isinstance(getattr(risk, k), set) else eval(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "score"])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--set", action="append", default=[], help="NAME=VALUE: параметр src/risk.py")
    ap.add_argument("--gt", default=None, help="разметка: посчитать Score B через evaluate.py")
    ap.add_argument("--top", type=int, default=5, help="сколько самых длинных алармов показать")
    args = ap.parse_args()
    videos = list_videos(Path(args.videos))

    if args.cmd == "dump":
        for v in videos:
            dump(v)
        return

    apply_overrides(args.set)
    pred, tot_frames, tot_hi, tot_alarms, tot_dur = {"videos": {}}, 0, 0, 0, 0.0
    for v in videos:
        path = CACHE / f"{v.stem}.pkl.gz"
        if not path.exists():
            print(f"[{v.name}] нет кэша — сначала: risk_replay.py dump --videos {v}")
            continue
        with gzip.open(path, "rb") as f:
            cached = pickle.load(f)
        if "width" not in cached:                 # кэш старого формата
            cap = cv2.VideoCapture(str(v))
            cached["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cached["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        curve = replay(cached)
        al = alarms(curve)
        hi = sum(s >= THETA for _, s in curve)
        dur = len(curve) / cached["fps"]
        tot_frames, tot_hi, tot_alarms, tot_dur = tot_frames + len(curve), tot_hi + hi, tot_alarms + len(al), tot_dur + dur
        print(f"[{v.name}] >=0.5: {hi / len(curve):6.1%} кадров, алармов {len(al):3d} "
              f"({len(al) / dur * 60:.1f}/мин)")
        for a in sorted(al, key=lambda a: a[0] - a[1])[:args.top]:
            print(f"      {a[0]:7.1f}-{a[1]:7.1f}s  ({a[1] - a[0]:5.1f}s, max {a[2]:.2f})")
        pred["videos"][v.name] = {"events": [], "risk": curve}
    if tot_frames:
        print(f"ВСЕГО: >=0.5 {tot_hi / tot_frames:.1%} кадров, {tot_alarms} алармов "
              f"({tot_alarms / tot_dur * 60:.2f}/мин)")
    if args.gt:
        import evaluate
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        rep = evaluate.evaluate(gt, pred)
        print(json.dumps({k: rep[k] for k in rep if "B" in k or "alarm" in k.lower() or "AP" in k}, indent=1))


if __name__ == "__main__":
    main()
