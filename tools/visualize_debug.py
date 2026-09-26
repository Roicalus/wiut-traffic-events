"""visualize_debug.py — видео с разметкой поверх результата пайплайна (src/render.py).

Сабмит целиком, как его увидит жюри: события и риск — из predictions.json
(run_submission.py), боксы/светофор/зоны — из кэша наблюдений tools/dev_loop.py
(того же прохода Part A), пара объектов, дающая риск, — из кэша Part B
(tools/risk_replay.py dump), если он есть:

    python tools/dev_loop.py --videos samples                         # один раз, строит cache/
    python tools/visualize_debug.py --video samples/C3896.MP4 --predictions predictions_samples.json

Без --predictions события считаются текущими правилами из кэша (все классы,
удобно при правке правил). Без кэша — отдельный проход pipeline.extract
(медленно). --all-classes — подсвечивать боксы и экспериментальных классов.

Результат — debug/<видео>.debug.mp4 (H.264, играет в браузере и любом плеере).
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import solution  # noqa: E402
from src import align, render, risk  # noqa: E402
from src.pipeline import extract, get_zones, infer, load_obs, reference_view  # noqa: E402
from src.rules import compute_events_debug  # noqa: E402


def risk_explanations(video_path):
    """[(t, score, пара риска)] из кэша Part B — тем же RiskScorer и с той же
    задержкой на шаг, что в RiskEstimator. None — кэша нет."""
    path = ROOT / "cache" / "risk" / f"{Path(video_path).stem}.pkl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        cached = pickle.load(f)
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    y0 = int(h * risk.CROP_TOP_FRAC)
    scorer, out, pending = risk.RiskScorer((w, h - y0)), [], None
    for _idx, t, dets in cached["frames"]:
        if pending is not None:
            score = scorer.feed(pending[1], pending[0])
            if scorer.explain:
                out.append((t, score, dict(scorer.explain, y0=y0)))
        pending = (t, dets)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--predictions", default=None, help="predictions.json из run_submission.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--no-cache", action="store_true", help="не брать кэш dev_loop, посчитать проход заново")
    ap.add_argument("--start", type=float, default=0.0, help="рендерить с этой секунды")
    ap.add_argument("--end", type=float, default=None, help="рендерить до этой секунды")
    ap.add_argument("--all-classes", action="store_true",
                    help="подсвечивать боксы и экспериментальных классов (по умолчанию — только solution.CLASSES)")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else ROOT / "debug" / f"{video_path.stem}.debug.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = ROOT / "cache" / f"{video_path.name}.obs.pkl.gz"
    if cache.exists() and not args.no_cache:
        obs = load_obs(cache)
        # треки в кэше не зависят от зон, а зоны могли поменяться после записи кэша
        # (новая зона в zones.json) — совмещаем текущие с этим роликом, как extract()
        obs.zones, obs.extra["align"] = align.aligned_zones(video_path, get_zones())
        print(f"Кэш наблюдений: {cache}; зоны: {obs.extra.get('align')}")
    else:
        print("Кэша нет — считаю проход Part A (медленно)...")
        obs = extract(str(video_path))

    if args.predictions:
        entry = json.loads(Path(args.predictions).read_text())["videos"].get(video_path.name)
        if entry is None:
            raise SystemExit(f"{video_path.name} нет в {args.predictions}")
        events, risk_curve = entry["events"], entry.get("risk", [])
    else:
        events, risk_curve = infer(obs, classes=None), []

    records, zones_ref = reference_view(obs)
    debug_events = compute_events_debug(records, zones_ref, obs.light_samples)
    if not args.all_classes:
        shown = set(solution.CLASSES) | set(solution.DIAGNOSTIC_CLASSES)
        debug_events = [ev for ev in debug_events if ev[2] in shown]
    # диагностические классы не попадают в predictions.json — на таймлайн из тех же правил
    events = list(events) + [[s, e, lbl] for s, e, lbl, _ in debug_events if lbl in solution.DIAGNOSTIC_CLASSES]
    stride = 1 if obs.meta.get("stride_changes") else obs.meta.get("stride", 3)
    explains = risk_explanations(video_path) if risk_curve else None
    print(f"Событий: {len(events)}, привязанных к объектам: {len(debug_events)}, "
          f"risk-сэмплов: {len(risk_curve)}, пар риска: {len(explains or [])}")
    render.render(video_path, out_path, obs.zones, obs.records, obs.light_samples, events, risk_curve,
                  debug_events, stride, explains=explains, max_width=args.max_width,
                  start=args.start, end=args.end)
    print(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
