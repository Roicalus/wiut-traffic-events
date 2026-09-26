"""Stitching track fragments after an occlusion (a pole, another car, etc.).

If track A ended close in place and time to the start of track B of the same
class, they are treated as one object. Needed because even with a long
tracker buffer, a full occlusion by a pole for several seconds can give
the very same stationary car a new track_id.

Run:
    python src/stitch_tracks.py --tracks src/tracks/C3896.json

Adds a "stitched_id" (int) field to every object — use it instead of
"track_id" in rules where the object's continuous history matters (stopped_vehicle
etc.). The original "track_id" is left untouched.
"""
import argparse
import json
from pathlib import Path

MAX_GAP_SEC = 5.0     # max time gap between the end of A and the start of B
MAX_DIST_PX = 150.0   # max distance between centres (in pixels of the original resolution)


def center(rec):
    return ((rec["x1"] + rec["x2"]) / 2, (rec["y1"] + rec["y2"]) / 2)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


PERSON_CLS = 0
SPLIT_ID_OFFSET = 10_000_000   # ids of split-off parts do not collide with tracker ids


def _family(cls):
    return "person" if cls == PERSON_CLS else "vehicle"


def split_class_switches(records):
    """ByteTrack in ultralytics is class-agnostic: a pedestrian whose box is covered
    by a car may "hand over" its track_id to it (C3897, #829: a person until 301.9 s,
    then the same id on a car — and the car became a "pedestrian on the roadway").
    Such a track is split: records of the dominant family (person / vehicle)
    keep the id, the rest get a separate one. Sets r["obj_id"]."""
    counts = {}
    for r in records:
        c = counts.setdefault(r["track_id"], {"person": 0, "vehicle": 0})
        c[_family(r["cls"])] += 1
    for r in records:
        c = counts[r["track_id"]]
        main = "person" if c["person"] > c["vehicle"] else "vehicle"
        r["obj_id"] = r["track_id"] if _family(r["cls"]) == main else SPLIT_ID_OFFSET + r["track_id"]
    return records


def build_track_summaries(records):
    by_id = {}
    for r in records:
        by_id.setdefault(r.get("obj_id", r["track_id"]), []).append(r)
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
    """Greedily stitches tracks: sort by start time, and for each one
    try to find the best "previous" track of the same class that
    ended shortly before and nearby."""
    ordered = sorted(summaries.values(), key=lambda s: s["start_t"])
    parent = {s["track_id"]: s["track_id"] for s in ordered}  # union-find

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # "open tail" candidates per class: list of (end_t, end_c, track_id)
    open_tails = {}  # cls -> list of dict(end_t, end_c, root_id)

    for s in ordered:
        cls = s["cls"]
        cands = open_tails.get(cls, [])
        best = None
        best_cost = None
        for tail in cands:
            gap = s["start_t"] - tail["end_t"]
            if gap <= 0 or gap > MAX_GAP_SEC:   # 0 — both objects are in the same frame, not a continuation
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
    """Wrapper without file I/O: takes records from track.run_tracker(),
    returns the same records with a 'stitched_id' field added to each
    record (mutates the list in place and returns it for convenience)."""
    global MAX_GAP_SEC, MAX_DIST_PX
    prev_gap, prev_dist = MAX_GAP_SEC, MAX_DIST_PX
    MAX_GAP_SEC, MAX_DIST_PX = max_gap_sec, max_dist_px
    try:
        split_class_switches(records)
        summaries = build_track_summaries(records)
        mapping = stitch(summaries)
        for r in records:
            r["stitched_id"] = mapping[r["obj_id"]]
    finally:
        MAX_GAP_SEC, MAX_DIST_PX = prev_gap, prev_dist
    return records


def main():
    global MAX_GAP_SEC, MAX_DIST_PX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="path to the JSON from track.py")
    ap.add_argument("--out", default=None, help="output path (default: alongside, .stitched.json)")
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
    print(f"Tracks before: {n_before}, after stitching: {n_after} "
          f"({n_before - n_after} fragments merged)")

    data["meta"]["stitch_max_gap_sec"] = MAX_GAP_SEC
    data["meta"]["stitch_max_dist_px"] = MAX_DIST_PX
    out_path.write_text(json.dumps(data, indent=1))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
