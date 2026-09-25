"""dev_loop.py — быстрый цикл калибровки Part A на своей разметке.

1-й запуск: гоняет дорогой проход (YOLO+трекер+светофор+сканер) по каждому
видео и кэширует наблюдения в --cache. Дальше правила/постпроцессинг
пересчитываются из кэша за секунды — правьте пороги в src/rules.py или
src/postprocess.py и перезапускайте.

    python tools/dev_loop.py --videos samples --gt my_labels.json
    python tools/dev_loop.py --videos samples --gt my_labels.json --ablate
    python tools/dev_loop.py --videos samples                 # без разметки: счётчики

Выводит:
  * сколько сегментов каждого класса нашлось (все классы, даже выключенные) —
    для "теста тишины": класс, которого в сэмплах нет, не должен срабатывать;
  * официальный отчёт evaluate.py для текущего solution.CLASSES;
  * --ablate: как меняется Score A, если добавить к CLASSES каждый из
    экспериментальных классов по отдельности (решение, что включать).

После изменения src/track.py (детектор/трекер/stride) — удалите кэш (--refresh).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
import solution  # noqa: E402
from src.pipeline import extract, infer, load_obs, save_obs  # noqa: E402


def score(gt, videos_events, classes):
    pred = {"videos": {v: {"events": [e for e in evs if e[2] in classes]}
                       for v, evs in videos_events.items()}}
    return evaluate.evaluate(gt, pred, per_video=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="папка с .mp4 или один файл")
    ap.add_argument("--gt", default=None, help="my_labels.json (tools/build_ground_truth.py)")
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--refresh", action="store_true", help="пересчитать кэш наблюдений")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--out", default=None, help="записать predictions (текущие CLASSES) сюда")
    ap.add_argument("--list", default=None,
                    help="через запятую: классы, чьи события напечатать с временем (или all)")
    args = ap.parse_args()

    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    cache = Path(args.cache)

    all_events = {}
    for path in videos:
        cpath = cache / f"{path.name}.obs.pkl.gz"
        if cpath.exists() and not args.refresh:
            obs = load_obs(cpath)
        else:
            t0 = time.perf_counter()
            obs = extract(str(path))
            el = time.perf_counter() - t0
            print(f"[{path.name}] extract {el:.0f}s = {el / max(obs.duration, 1e-6):.2f}x длительности "
                  f"(stride_changes={obs.meta.get('stride_changes')})")
            save_obs(obs, cpath)
        t0 = time.perf_counter()
        all_events[path.name] = infer(obs, classes=None)
        cnt = Counter(e[2] for e in all_events[path.name])
        print(f"[{path.name}] infer {time.perf_counter() - t0:.1f}s  " +
              ", ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
        if args.list:
            wanted = None if args.list == "all" else set(args.list.split(","))
            for st, en, lbl in all_events[path.name]:
                if wanted is None or lbl in wanted:
                    print(f"      {lbl:<20} {st:8.2f} - {en:8.2f}  ({en - st:6.1f} s)"
                          f"   {int(st // 60)}:{st % 60:05.2f}")

    if args.out:
        out = {"team": "dev", "videos": {v: {"events": [e for e in evs if e[2] in solution.CLASSES],
                                            "risk": []} for v, evs in all_events.items()}}
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")

    if not args.gt:
        return
    gt = json.loads(Path(args.gt).read_text())
    gt = {k: v for k, v in gt.items() if k in all_events}
    print(f"\n==== CLASSES = {solution.CLASSES}")
    rep = score(gt, all_events, set(solution.CLASSES))
    evaluate.print_report(rep)

    if args.ablate:
        base = rep["part_a"]["score_a"]
        print(f"\n==== ablation (база Score A = {base:.4f})")
        for c in solution.EXPERIMENTAL_CLASSES:
            if c in solution.CLASSES:
                continue
            s = score(gt, all_events, set(solution.CLASSES) | {c})["part_a"]["score_a"]
            n = sum(1 for evs in all_events.values() for e in evs if e[2] == c)
            print(f"  + {c:<20} pred={n:<4} Score A = {s:.4f}  ({s - base:+.4f})")
        for c in solution.CLASSES:
            s = score(gt, all_events, set(solution.CLASSES) - {c})["part_a"]["score_a"]
            print(f"  - {c:<20}          Score A = {s:.4f}  ({s - base:+.4f})")


if __name__ == "__main__":
    main()
