"""analyze.py — полный разбор одного ролика для живого демо (demo/app.py).

Тот же код, что в сабмите (pipeline.extract/infer, RiskEstimator), плюс то,
что нужно посетителю сайта: видео с разметкой, таймлайн, кривая риска, JSON.

На CPU (бесплатный хостинг) 4K декодируется медленно, поэтому ролик сначала
пережимается в DEMO_WIDTH x ... при DEMO_FPS (ffmpeg из imageio-ffmpeg), а
детектор идёт на меньшем входе. Правила работают в пикселях опорного 4K-кадра
(pipeline.to_reference_pixels), так что их пороги те же, что в сабмите;
время событий — в секундах исходного ролика.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import cv2

import solution
from src import pipeline, render, risk
from src.rules import compute_events_debug

# Настройки под железо хостинга — переменными окружения, без правки кода.
MAX_DURATION_SEC = float(os.environ.get("DEMO_MAX_SEC", 150))    # до 2.5 минут
DEMO_WIDTH = int(os.environ.get("DEMO_WIDTH", 1920))
DEMO_FPS = float(os.environ.get("DEMO_FPS", 15))
DEMO_TRACKER = {"imgsz": int(os.environ.get("DEMO_IMGSZ", 960)), "stride": 3}   # 5 Гц при 15 fps
DEMO_RISK_STRIDE = 3                          # каждый 3-й кадр — детектор Part B (внутри — свой stride)


class VideoError(ValueError):
    """Ролик не подходит (не читается, слишком длинный); текст — для посетителя сайта."""


def probe(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("Cannot open the video. Please upload an .mp4 (H.264/H.265).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or n <= 0 or w <= 0:
        raise VideoError("The video has no frames or no frame rate.")
    return {"fps": fps, "n_frames": n, "width": w, "height": h, "duration": n / fps}


def transcode(src, dst, info, progress=None) -> Path:
    """Уменьшаем до DEMO_WIDTH и DEMO_FPS, если ролик больше. Без звука."""
    if info["width"] <= DEMO_WIDTH and info["fps"] <= DEMO_FPS + 0.5:
        return Path(src)
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-an",
           "-vf", f"scale={min(DEMO_WIDTH, info['width'])}:-2,fps={DEMO_FPS}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, errors="replace")
    for line in proc.stderr:            # ffmpeg пишет прогресс в stderr: time=00:00:12.34
        m = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
        if m and progress is not None:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            progress(min(t / info["duration"], 1.0))
    if proc.wait() != 0 or not Path(dst).exists():
        raise VideoError("Could not re-encode the video.")
    return Path(dst)


def risk_pass(path, progress=None) -> tuple[list, list]:
    """Part B так же, как в харнессе: кадры по порядку, reset/step. Детектор
    вызывается на каждом DEMO_RISK_STRIDE-м кадре (как --risk-stride), в
    остальных кадрах держится последний скор."""
    info = probe(path)
    est = risk.RiskEstimator()
    est.reset({"video_id": Path(path).name, "fps": info["fps"], "width": info["width"],
               "height": info["height"], "n_frames": info["n_frames"]})
    est.explain_log = []
    cap = cv2.VideoCapture(str(path))
    curve, idx, last = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / info["fps"]
        if idx % DEMO_RISK_STRIDE == 0:
            last = float(est.step(frame, t))
        curve.append([round(t, 3), round(last, 4)])
        idx += 1
        if progress is not None and idx % 30 == 0:
            progress(idx / info["n_frames"])
    cap.release()
    y0 = int(info["height"] * risk.CROP_TOP_FRAC)
    explains = [(t, s, dict(e, y0=y0)) for t, s, e in est.explain_log if e]
    return curve, explains


def prepare(video_path, work_dir, progress=None) -> tuple[Path, dict]:
    """CPU: проверка и пережатие. Возвращает (путь к рабочему ролику, info)."""
    say = progress or (lambda frac, msg: None)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    info = probe(video_path)
    if info["duration"] > MAX_DURATION_SEC:
        raise VideoError(f"The clip is {info['duration']:.0f} s long; the demo accepts up to {MAX_DURATION_SEC:.0f} s.")
    say(0.0, "Preparing the video")
    path = transcode(video_path, work / "input.mp4", info, lambda f: say(0.25 * f, "Preparing the video"))
    return path, info


def compute(path, progress=None) -> dict:
    """Детектор и правила: Part A и Part B. На ZeroGPU — внутри @spaces.GPU,
    поэтому возвращает только сериализуемое."""
    say = progress or (lambda frac, msg: None)
    t1 = time.perf_counter()
    say(0.25, "Part A: detector, tracker, traffic light")
    obs = pipeline.extract(str(path), classes=solution.CLASSES, tracker_kwargs=DEMO_TRACKER)
    events = pipeline.infer(obs, classes=solution.CLASSES)
    records, zones_ref = pipeline.reference_view(obs)
    shown = list(solution.CLASSES) + list(solution.DIAGNOSTIC_CLASSES)
    debug_events = compute_events_debug(records, zones_ref, obs.light_samples, classes=shown)
    t2 = time.perf_counter()
    say(0.55, "Part B: accident risk")
    curve, explains = risk_pass(path, lambda f: say(0.55 + 0.2 * f, "Part B: accident risk"))
    obs.scanner = None
    return {"obs": obs, "events": events, "debug_events": debug_events, "risk": curve,
            "explains": explains, "device": str(pipeline.track._pick_device()),
            "timings": {"part_a": t2 - t1, "part_b": time.perf_counter() - t2}}


def finish(video_path, path, info, work_dir, computed, progress=None) -> dict:
    """CPU: видео с разметкой и events.json."""
    say = progress or (lambda frac, msg: None)
    work = Path(work_dir)
    obs, events, debug_events = computed["obs"], computed["events"], computed["debug_events"]
    diag = [[s, e, lbl] for s, e, lbl, _ in debug_events if lbl in solution.DIAGNOSTIC_CLASSES]
    t3 = time.perf_counter()
    say(0.75, "Rendering the annotated video")
    annotated = render.render(path, work / "annotated.mp4", obs.zones, obs.records, obs.light_samples,
                              events + diag, computed["risk"], debug_events, DEMO_TRACKER["stride"],
                              explains=computed["explains"],
                              progress=lambda f: say(0.75 + 0.24 * f, "Rendering the annotated video"))
    light = [s for _, s in obs.light_samples] if obs.light_samples else []
    timings = dict(computed["timings"], render=time.perf_counter() - t3)
    result = {
        "video": Path(video_path).name,
        "info": {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()},
        "events": sorted(events),
        "diagnostics": sorted(diag),
        "risk": computed["risk"],
        "align": obs.extra.get("align"),
        "light": {s: round(light.count(s) / len(light), 3) for s in set(light)} if light else None,
        "device": computed["device"],
        "timings_sec": {k: round(v, 1) for k, v in timings.items()},
    }
    json_path = work / "events.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    say(1.0, "Done")
    return dict(result, annotated=str(annotated), json=str(json_path))


def analyze(video_path, work_dir, progress=None) -> dict:
    """Все три этапа подряд (локально, без ZeroGPU). progress(frac, message)."""
    t0 = time.perf_counter()
    path, info = prepare(video_path, work_dir, progress)
    t_prep = time.perf_counter() - t0
    computed = compute(path, progress)
    result = finish(video_path, path, info, work_dir, computed, progress)
    result["timings_sec"].update(transcode=round(t_prep, 1), total=round(time.perf_counter() - t0, 1))
    return result
