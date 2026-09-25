"""budget.py — общий секундомер Part A + Part B на одно видео.

Харнесс засекает время ДО detect_events() и считает бюджет
TIME_FACTOR x duration на обе части вместе; превышение = видео пустое
(теряется и Part A). Part B идёт вторым и не знает, сколько съела Part A,
поэтому detect_events() отмечает старт здесь, а RiskEstimator по этой
отметке считает, сколько времени у него осталось, и сам разрежает
инференс, если не успевает.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

TIME_FACTOR = 3.0        # как в run_submission.py
SAFETY = 0.85            # целимся в 85% бюджета — запас на джиттер машины организаторов
PART_A_SHARE = 1.5       # цель для прохода трекера: <= 1.5 x duration

# Только для локальной разработки на медленном железе (CPU/встройка):
# WIUT_TIME_GUARD=0 выключает оба предохранителя, чтобы результат не
# отличался от GPU-прогона из-за увеличенного stride. Официальный прогон
# переменную не задаёт -> предохранители включены.
GUARD_ON = os.environ.get("WIUT_TIME_GUARD", "1") != "0"

_starts: dict[str, float] = {}


def mark_start(video_path: str) -> None:
    _starts[Path(video_path).name] = time.perf_counter()


def deadline_for(video_id: str, duration: float) -> float:
    """Абсолютный perf_counter-дедлайн для видео (с запасом SAFETY).
    Если detect_events не вызывался (например, dev-прогон только Part B),
    отсчитываем от текущего момента."""
    if not GUARD_ON:
        return float("inf")
    start = _starts.get(video_id, time.perf_counter())
    return start + TIME_FACTOR * duration * SAFETY
