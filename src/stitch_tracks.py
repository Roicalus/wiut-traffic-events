"""Сшивание обрывков треков после перекрытия (столб, другая машина и т.д.).

Если трек A закончился рядом по месту и времени с началом трека B того же
класса — считаем их одним объектом. Нужно, потому что даже с длинным
буфером трекера полное перекрытие столбом на несколько секунд может дать
новый track_id той же самой стоящей машине.

Запуск:
    python src/stitch_tracks.py --tracks src/tracks/C3896.json

Добавляет каждому объекту поле "stitched_id" (int) — используй его вместо
"track_id" в правилах, где важна непрерывная история объекта (stopped_vehicle
и т.п.). Исходный "track_id" не трогается.
"""
import argparse
import json
from pathlib import Path

MAX_GAP_SEC = 5.0     # макс. разрыв по времени между концом A и началом B
MAX_DIST_PX = 150.0   # макс. расстояние между центрами (в пикселях исходного разрешения)


def center(rec):
    return ((rec["x1"] + rec["x2"]) / 2, (rec["y1"] + rec["y2"]) / 2)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def build_track_summaries(records):
    by_id = {}
    for r in records:
        by_id.setdefault(r["track_id"], []).append(r)
    summaries = {}
    for tid, recs in by_id.items():
        recs.sort(key=lambda r: r["frame"])
        summaries[tid] = {
            "track_id": tid,
            "cls": recs[0]["cls"],
            "start_t": recs[0]["t_sec"],
            "end_t": recs[-1]["t_sec"],
            "start_c": center(recs[0]),
            "end_c": center(recs[-1]),
            "records": recs,
        }
    return summaries


def stitch(summaries):
    """Жадно сшивает треки: сортируем по времени начала, для каждого
    пытаемся найти лучший "предыдущий" трек того же класса, который
    закончился незадолго до и рядом."""
    ordered = sorted(summaries.values(), key=lambda s: s["start_t"])
    parent = {s["track_id"]: s["track_id"] for s in ordered}  # union-find

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # кандидаты в "открытые хвосты" по классу: list of (end_t, end_c, track_id)
    open_tails = {}  # cls -> list of dict(end_t, end_c, root_id)

    for s in ordered:
        cls = s["cls"]
        cands = open_tails.get(cls, [])
        best = None
        best_cost = None
        for tail in cands:
            gap = s["start_t"] - tail["end_t"]
            if gap <= 0 or gap > MAX_GAP_SEC:   # 0 — оба объекта в одном кадре, это не продолжение
                continue
            d = dist(tail["end_c"], s["start_c"])
            if d > MAX_DIST_PX:
                continue
            cost = gap / MAX_GAP_SEC + d / MAX_DIST_PX
            if best is None or cost < best_cost:
                best, best_cost = tail, cost

        if best is not None:
            root = find(best["root_id"])
            parent[find(s["track_id"])] = root
            cands.remove(best)
        else:
            root = find(s["track_id"])

        cands.append({"end_t": s["end_t"], "end_c": s["end_c"], "root_id": root})
        open_tails[cls] = cands

    return {tid: find(tid) for tid in parent}


def stitch_records(records, max_gap_sec=MAX_GAP_SEC, max_dist_px=MAX_DIST_PX):
    """Обёртка без файлового I/O: принимает records из track.run_tracker(),
    возвращает те же records с добавленным полем 'stitched_id' на каждой
    записи (мутирует список на месте и возвращает его же для удобства)."""
    global MAX_GAP_SEC, MAX_DIST_PX
    prev_gap, prev_dist = MAX_GAP_SEC, MAX_DIST_PX
    MAX_GAP_SEC, MAX_DIST_PX = max_gap_sec, max_dist_px
    try:
        summaries = build_track_summaries(records)
        mapping = stitch(summaries)
        for r in records:
            r["stitched_id"] = mapping[r["track_id"]]
    finally:
        MAX_GAP_SEC, MAX_DIST_PX = prev_gap, prev_dist
    return records


def main():
    global MAX_GAP_SEC, MAX_DIST_PX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="путь к JSON из track.py")
    ap.add_argument("--out", default=None, help="куда писать (по умолчанию рядом, .stitched.json)")
    ap.add_argument("--max-gap-sec", type=float, default=MAX_GAP_SEC)
    ap.add_argument("--max-dist-px", type=float, default=MAX_DIST_PX)
    args = ap.parse_args()

    MAX_GAP_SEC = args.max_gap_sec
    MAX_DIST_PX = args.max_dist_px

    in_path = Path(args.tracks)
    out_path = Path(args.out) if args.out else in_path.with_suffix(".stitched.json")

    data = json.loads(in_path.read_text())
    records = data["tracks"]

    summaries = build_track_summaries(records)
    mapping = stitch(summaries)

    for r in records:
        r["stitched_id"] = mapping[r["track_id"]]

    n_before = len(summaries)
    n_after = len(set(mapping.values()))
    print(f"Треков было: {n_before}, после сшивки: {n_after} "
          f"(объединено {n_before - n_after} обрывков)")

    data["meta"]["stitch_max_gap_sec"] = MAX_GAP_SEC
    data["meta"]["stitch_max_dist_px"] = MAX_DIST_PX
    out_path.write_text(json.dumps(data, indent=1))
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
