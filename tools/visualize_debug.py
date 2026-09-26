"""visualize_debug.py — video with annotations over the pipeline output (src/render.py).

The whole submission as the jury will see it: events and risk come from predictions.json
(run_submission.py), boxes/traffic light/zones from the tools/dev_loop.py observation cache
(the same Part A pass), the object pair that produces the risk from the Part B cache
(tools/risk_replay.py dump), if present:

    python tools/dev_loop.py --videos samples                         # once, builds cache/
    python tools/visualize_debug.py --video samples/C3896.MP4 --predictions predictions_samples.json

Without --predictions, events are computed by the current rules from the cache (all classes,
handy when editing rules). Without a cache — a separate pipeline.extract pass
(slow). --all-classes — also highlight boxes of experimental classes.

Output: debug/<video>.debug.mp4 (H.264, plays in a browser and any player).
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
    """[(t, score, risk pair)] from the Part B cache — with the same RiskScorer and the same
    one-step delay as in RiskEstimator. None — no cache."""
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
    ap.add_argument("--predictions", default=None, help="predictions.json from run_submission.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--no-cache", action="store_true", help="do not use the dev_loop cache, recompute the pass")
    ap.add_argument("--start", type=float, default=0.0, help="render from this second")
    ap.add_argument("--end", type=float, default=None, help="render up to this second")
    ap.add_argument("--all-classes", action="store_true",
                    help="also highlight boxes of experimental classes (default: only solution.CLASSES)")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else ROOT / "debug" / f"{video_path.stem}.debug.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = ROOT / "cache" / f"{video_path.name}.obs.pkl.gz"
    if cache.exists() and not args.no_cache:
        obs = load_obs(cache)
        # cached tracks do not depend on zones, but zones may have changed after the cache was written
        # (a new zone in zones.json) — align the current ones to this clip, as extract() does
        obs.zones, obs.extra["align"] = align.aligned_zones(video_path, get_zones())
        print(f"Observation cache: {cache}; zones: {obs.extra.get('align')}")
    else:
        print("No cache — running the Part A pass (slow)...")
        obs = extract(str(video_path))

    if args.predictions:
        entry = json.loads(Path(args.predictions).read_text())["videos"].get(video_path.name)
        if entry is None:
            raise SystemExit(f"{video_path.name} is not in {args.predictions}")
        events, risk_curve = entry["events"], entry.get("risk", [])
    else:
        events, risk_curve = infer(obs, classes=None), []

    records, zones_ref = reference_view(obs)
    debug_events = compute_events_debug(records, zones_ref, obs.light_samples)
    if not args.all_classes:
        shown = set(solution.CLASSES) | set(solution.DIAGNOSTIC_CLASSES)
        debug_events = [ev for ev in debug_events if ev[2] in shown]
    # diagnostic classes do not go into predictions.json — they reach the timeline from the same rules
    events = list(events) + [[s, e, lbl] for s, e, lbl, _ in debug_events if lbl in solution.DIAGNOSTIC_CLASSES]
    stride = 1 if obs.meta.get("stride_changes") else obs.meta.get("stride", 3)
    explains = risk_explanations(video_path) if risk_curve else None
    print(f"Events: {len(events)}, tied to objects: {len(debug_events)}, "
          f"risk samples: {len(risk_curve)}, risk pairs: {len(explains or [])}")
    render.render(video_path, out_path, obs.zones, obs.records, obs.light_samples, events, risk_curve,
                  debug_events, stride, explains=explains, max_width=args.max_width,
                  start=args.start, end=args.end)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
