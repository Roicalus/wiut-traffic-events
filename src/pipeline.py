"""pipeline.py — Part A в две стадии:

  extract(video)  -> Observations   дорого: ОДИН проход декодирования 4K,
                                    в нём YOLO+ByteTrack, светофор и
                                    obstacle/fire-сканер (колбэки on_frame)
  infer(obs)      -> events         дёшево: сшивка треков, правила,
                                    постпроцессинг

Разделение нужно для разработки: наблюдения кэшируются на диск
(save_obs/load_obs), и подбор порогов правил против своей разметки идёт
за секунды, а не по 8 минут YOLO на ролик (см. tools/dev_loop.py).
В сабмишене solution.detect_events() просто вызывает обе стадии подряд.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src import align, budget, track
from src.light_state import LightScanner
from src.obstacle_fire import ObstacleFireScanner
from src.postprocess import postprocess
from src.rules import compute_events, load_zones
from src.stitch_tracks import stitch_records

REPO_ROOT = Path(__file__).resolve().parent.parent
ZONES_PATH = REPO_ROOT / "zones.json"

_zones_cache: dict | None = None
_model_cache = None


def get_zones() -> dict:
    global _zones_cache
    if _zones_cache is None:
        if ZONES_PATH.exists():
            _zones_cache = load_zones(ZONES_PATH)
        else:
            print(f"WARNING: {ZONES_PATH} не найден — классы на зонах не детектируются")
            _zones_cache = {}
    return _zones_cache


def get_model():
    global _model_cache
    if _model_cache is None:
        from ultralytics import YOLO
        _model_cache = YOLO(track.DEFAULT_MODEL)
    return _model_cache


def warm_up() -> None:
    """Всё, что стоит одинаково для любого видео: загрузка весов, инициализация
    CUDA и первый инференс (подбор ядер), признаки опорного кадра зон.
    Вызывается при импорте solution.py — харнесс импортирует решение ДО
    старта секундомера видео. Иначе эти ~10 с съедал бы бюджет первого
    ролика: 3-секундный клип (бюджет 9 с) засчитывался пустым."""
    import numpy as np
    from src.risk import RiskEstimator
    dummy = np.zeros((360, 640, 3), np.uint8)
    device = track._pick_device()
    get_model().predict(dummy, imgsz=track.WARMUP_IMGSZ, device=device,
                        half=device != "cpu", verbose=False)
    RiskEstimator._get_model().predict(dummy, imgsz=640, device=device,
                                       half=device != "cpu", verbose=False)
    align.load_reference()
    get_zones()


@dataclass
class Observations:
    meta: dict
    records: list
    light_samples: list | None
    scanner: ObstacleFireScanner | None
    duration: float = 0.0
    extra: dict = field(default_factory=dict)
    zones: dict | None = None      # зоны, совмещённые с ЭТИМ видео (см. src/align.py)


OBSTACLE_FIRE_CLASSES = {"road_obstacle", "fire_smoke"}


def extract(video_path: str, time_budget_sec: float | None = None, classes=None,
            tracker_kwargs: dict | None = None) -> Observations:
    """classes — какие классы потом понадобятся (None — все, для dev-кэша):
    сканер obstacle/fire (MOG2 + HSV на каждом кадре) создаётся, только если
    нужны его классы. tracker_kwargs — переопределения track.run_tracker
    (демо на CPU: imgsz/stride); в сабмите не задаются."""
    zones, align_report = align.aligned_zones(video_path, get_zones())
    name = Path(video_path).name
    if align_report.get("status") == "ok":
        print(f"[{name}] зоны совмещены ({align_report['model']}, опорный кадр {align_report['ref']}): "
              f"dx={align_report['dx']} dy={align_report['dy']} px, rot={align_report['rot_deg']}°, "
              f"scale={align_report['scale']}, inliers={align_report['inliers']}"
              + (f" — ВНИМАНИЕ: {align_report['warning']}" if align_report.get("warning") else ""))
    else:
        print(f"[{name}] WARNING: зоны НЕ совмещены ({align_report}) — используются как есть")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n_frames / fps if fps else 0.0

    light = LightScanner(zones, full_height=full_height) if "light_roi" in zones else None
    scanner = None
    if classes is None or OBSTACLE_FIRE_CLASSES & set(classes):
        scanner = ObstacleFireScanner(zones, fps, full_height=full_height, progress=False)

    def on_frame(cropped, t_sec, result):
        if light is not None:
            light.on_tracker_frame(cropped, t_sec, result)
        if scanner is not None:
            scanner.on_tracker_frame(cropped, t_sec, result)

    if time_budget_sec is None and budget.GUARD_ON:
        time_budget_sec = budget.PART_A_SHARE * duration
    meta, records = track.run_tracker(video_path, model=get_model(), on_frame=on_frame,
                                      time_budget_sec=time_budget_sec, **(tracker_kwargs or {}))
    return Observations(meta=meta, records=records,
                        light_samples=light.samples() if light is not None else None,
                        scanner=scanner, duration=duration,
                        extra={"align": align_report}, zones=zones)


def to_reference_pixels(records: list, zones: dict, width: float) -> tuple[list, dict]:
    """Правила и склейка треков работают в пикселях опорного кадра (4K):
    пороги вроде stitch_tracks.MAX_DIST_PX заданы в них. Видео другого
    разрешения (демо — 1920 px) приводится к этому масштабу; 4K — без изменений."""
    refs = align.load_reference()
    ref_w = refs[0]["full_width"] if refs else width
    k = ref_w / width if width else 1.0
    if abs(k - 1.0) < 1e-3:
        return records, zones
    scaled = [dict(r, x1=r["x1"] * k, y1=r["y1"] * k, x2=r["x2"] * k, y2=r["y2"] * k) for r in records]
    return scaled, {n: z * k for n, z in zones.items()}


def reference_view(obs: Observations) -> tuple[list, dict]:
    """Записи треков и зоны в пикселях опорного кадра, со сшитыми треками.
    stitched_id копируется и в obs.records — визуализация подсвечивает боксы
    по тем же id, что у правил (склейка в разных масштабах могла бы дать
    разные id)."""
    zones = getattr(obs, "zones", None) or get_zones()   # старый кэш — без совмещения
    records, zones = to_reference_pixels(obs.records, zones, obs.meta.get("width") or 0)
    stitch_records(records)
    if records is not obs.records:
        for src, ref in zip(obs.records, records):
            src["stitched_id"] = ref["stitched_id"]
    return records, zones


def infer(obs: Observations, classes=None, post_params=None) -> list[list]:
    records, zones = reference_view(obs)
    events = []
    try:
        events += compute_events(records, zones, obs.light_samples, classes)
    except Exception as exc:  # общая подготовка (annotate и т.п.) — у правил своя страховка
        print(f"[{obs.meta.get('video')}] compute_events упал: {exc!r}")
    if obs.scanner is not None and (classes is None or OBSTACLE_FIRE_CLASSES & set(classes)):
        try:
            events += obs.scanner.finish(obs.records)   # сканер — в пикселях самого видео
        except Exception as exc:
            print(f"[{obs.meta.get('video')}] obstacle_fire упал: {exc!r}")
    return postprocess(events, duration=obs.duration or None, classes=classes, params=post_params)


# ---------------------------------------------------------------- кэш (dev)
def save_obs(obs: Observations, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if obs.scanner is not None:
        obs.scanner._bg = None  # cv2 BackgroundSubtractor не сериализуется; finish() он не нужен
    with gzip.open(path, "wb") as f:
        pickle.dump(obs, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obs(path: Path) -> Observations:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
