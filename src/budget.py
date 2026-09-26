"""budget.py — a shared Part A + Part B stopwatch per video.

The harness starts the clock BEFORE detect_events() and counts a budget of
TIME_FACTOR x duration for both parts together; exceeding it = empty video
(Part A is lost too). Part B runs second and does not know how much Part A used,
so detect_events() records the start here, and RiskEstimator uses that mark
to compute how much time it has left and thins out inference itself
if it is falling behind.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

TIME_FACTOR = 3.0        # as in run_submission.py
SAFETY = 0.85            # aim for 85% of the budget — margin for jitter on the organizers' machine
PART_A_SHARE = 1.5       # target for the tracker pass: <= 1.5 x duration

# Only for local development on slow hardware (CPU/integrated GPU):
# WIUT_TIME_GUARD=0 disables both guards so the result does not
# differ from a GPU run because of an increased stride. The official run
# does not set the variable -> guards are on.
GUARD_ON = os.environ.get("WIUT_TIME_GUARD", "1") != "0"

_starts: dict[str, float] = {}


def mark_start(video_path: str) -> None:
    _starts[Path(video_path).name] = time.perf_counter()


def deadline_for(video_id: str, duration: float) -> float:
    """Absolute perf_counter deadline for the video (with the SAFETY margin).
    If detect_events was not called (e.g. a dev run of Part B only),
    count from the current moment."""
    if not GUARD_ON:
        return float("inf")
    start = _starts.get(video_id, time.perf_counter())
    return start + TIME_FACTOR * duration * SAFETY
