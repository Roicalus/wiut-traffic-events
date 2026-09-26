"""
solution.py — the hackathon interface. A thin wrapper: all logic lives in src/.

  Part A  detect_events(video_path)
          src/pipeline.py: ONE pass over the video (YOLO11s + ByteTrack; the traffic light
          and obstacle/fire are handled in the same pass), then rules on the tracks
          (src/rules.py) and segment post-processing (src/postprocess.py).
  Part B  RiskEstimator  (src/risk.py): its own causal YOLO11n pass,
          closest point of approach + required deceleration for pairs,
          and it watches the time budget itself.

Requires: zones.json and zones_ref.jpg/.json next to this file, weights/yolo11s.pt
and weights/yolo11n.pt (included in the repository; otherwise bash weights/download.sh).
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")         # no ultralytics network checks
os.environ.setdefault("YOLO_VERBOSE", "False")
# cuBLAS picks workspace memory and kernels based on GPU state: without a fixed
# workspace, fp16 convolution results may differ between runs (must be set before importing torch).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Logs are English now, but paths and exception texts may still contain non-ASCII: on a machine with
# a non-UTF-8 console, print() would raise UnicodeEncodeError inside detect_events and the video
# would be scored as empty.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent))  # src/ is importable from any cwd

import numpy as np  # noqa: E402

from src import budget  # noqa: E402
from src.risk import RiskEstimator  # noqa: E402,F401  (the harness looks for this name here)

random.seed(0)
np.random.seed(0)
try:
    import torch
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False      # cuDNN kernel autotuning is non-deterministic
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)  # no deterministic kernel -> warning, not error
except ImportError:
    pass

# Classes we ACTUALLY submit. Metric rule: a class we predicted that is absent
# from the test set is added to the mean with F1 = 0. So a class is enabled only
# if it has a decent F1 on our own labels of the samples, or, if the samples have
# none of it, if it almost never fires there ("silence test",
# tools/dev_loop.py prints counters for all classes).
CORE_CLASSES = ["congestion", "stopped_vehicle", "jaywalking", "red_light", "stop_line",
                "illegal_turn"]
# The rules exist and are computed, but are not submitted until they pass validation.
EXPERIMENTAL_CLASSES = ["wrong_way", "illegal_u_turn", "failure_to_yield",
                        "solid_line_crossing", "accident", "near_miss",
                        "road_obstacle", "fire_smoke"]

CLASSES: list[str] = list(CORE_CLASSES)
# Computed and shown in the visualization, but not submitted: there is no official
# class for it (mounting a curb island), and adding our own ids is not allowed.
DIAGNOSTIC_CLASSES = ["curb_mount"]

from src import pipeline  # noqa: E402

try:
    pipeline.warm_up()          # outside the video budget: the harness imports the solution before the stopwatch
except Exception as _exc:      # noqa: BLE001 — works without warm-up too, just slower
    print(f"WARNING: model warm-up failed: {_exc!r}")


def detect_events(video_path: str) -> list[list]:
    budget.mark_start(video_path)
    obs = pipeline.extract(video_path, classes=CLASSES)
    return pipeline.infer(obs, classes=CLASSES)
