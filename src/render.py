"""render.py — видео с разметкой поверх результата пайплайна.

Одна реализация для tools/visualize_debug.py (сэмплы, из кэша) и для
живого демо (src/analyze.py): зоны, боксы трекера, объект-нарушитель
(толстая рамка цвета нарушения), состояние светофора, плашка активных
событий, полоса риска Part B, пара объектов, дающая риск, и панель под
кадром — таймлайн всех событий по классам, кривая риска, бегунок.

Видео пишется в H.264 (imageio-ffmpeg, играет в браузере и любом плеере),
если его нет — в mp4v через OpenCV.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.rules import PERSON_CLASS, LightState

# Цвета по классам событий (BGR) — стабильные, чтобы глаз привыкал
EVENT_COLORS = {
    "congestion": (0, 165, 255),        # оранжевый
    "stopped_vehicle": (0, 0, 255),     # красный
    "jaywalking": (255, 0, 255),        # пурпурный
    "red_light": (0, 0, 139),           # тёмно-красный
    "stop_line": (255, 255, 0),         # голубой
    "accident": (0, 0, 200),
    "near_miss": (80, 80, 255),
    "wrong_way": (0, 100, 255),
    "illegal_u_turn": (180, 105, 255),
    "failure_to_yield": (147, 20, 255),
    "illegal_turn": (130, 0, 75),
    "solid_line_crossing": (0, 215, 255),
    "road_obstacle": (42, 42, 165),
    "fire_smoke": (0, 69, 255),
    "curb_mount": (0, 128, 128),      # диагностика: заезд на островок
}


# порядок важности: цвет рамки объекта с несколькими нарушениями — по первому
VIOLATION_PRIORITY = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
                      "illegal_turn", "failure_to_yield", "stopped_vehicle", "stop_line",
                      "solid_line_crossing", "jaywalking", "congestion", "road_obstacle", "fire_smoke",
                      "curb_mount"]


PANEL_ROW_H = 22
PANEL_LABEL_W = 150
ZONE_COLOR = (0, 200, 0)
LIGHT_COLORS = {"red": (0, 0, 255), "yellow": (0, 220, 220),
                 "green": (0, 255, 0), "unknown": (128, 128, 128)}


def draw_zones(frame, zones, scale):
    # только контуры: зоны покрывают почти весь кадр, заливка его зеленит
    for name, poly in zones.items():
        pts = (np.array(poly, dtype=np.float32) * scale).astype(int)
        cv2.polylines(frame, [pts], True, ZONE_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, ZONE_COLOR, 1, cv2.LINE_AA)


def draw_boxes(frame, recs_at_frame, scale, active_violators=None):
    """active_violators: dict stitched_id -> [нарушения, активные ПРЯМО СЕЙЧАС
    у этого объекта] (из compute_events_debug), самое важное первым — такой
    бокс рисуется толстой рамкой его цвета со всеми именами."""
    active_violators = active_violators or {}
    for r in recs_at_frame:
        x1, y1, x2, y2 = (int(r["x1"] * scale), int(r["y1"] * scale),
                           int(r["x2"] * scale), int(r["y2"] * scale))
        sid = r.get("stitched_id", r["track_id"])
        is_person = r["cls"] == PERSON_CLASS
        violations = active_violators.get(sid)
        if violations:
            color, thickness = EVENT_COLORS.get(violations[0], (0, 0, 255)), 3
            label = f"! {' + '.join(violations)} #{r['track_id']}->{sid}"
        else:
            color, thickness = ((255, 180, 0) if is_person else (0, 220, 0)), 2
            label = f"{r.get('cls_name', r['cls'])} #{r['track_id']}->{sid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def congestion_active_now(t_sec, debug_events):
    return any(lbl == "congestion" and s <= t_sec <= e for s, e, lbl, _oid in debug_events)


def in_queue_zone(r, zones_np):
    if "queue_zone" not in zones_np:
        return False
    cx, cy = (r["x1"] + r["x2"]) / 2.0, (r["y1"] + r["y2"]) / 2.0
    return cv2.pointPolygonTest(zones_np["queue_zone"], (cx, cy), False) >= 0


def draw_light(frame, zones, scale, state):
    if "light_roi" not in zones:
        return
    xs = [p[0] for p in zones["light_roi"]]
    ys = [p[1] for p in zones["light_roi"]]
    cx, cy = int(np.mean(xs) * scale), int(min(ys) * scale) - 15
    cv2.circle(frame, (cx, max(cy, 10)), 8, LIGHT_COLORS.get(state, (128, 128, 128)), -1)
    cv2.putText(frame, f"light: {state}", (cx + 15, max(cy, 10) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIGHT_COLORS.get(state, (128, 128, 128)), 1)


def draw_events(frame, w, h, active, recent):
    # верхняя плашка с активными сейчас событиями
    if active:
        text = " | ".join(f"{lbl}" for lbl in active)
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, "ACTIVE: " + text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    # тикер последних событий слева снизу
    for i, (s, e, lbl) in enumerate(recent[-6:]):
        color = EVENT_COLORS.get(lbl, (200, 200, 200))
        cv2.putText(frame, f"[{s:.1f}-{e:.1f}] {lbl}", (8, h - 15 - 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_risk(frame, w, risk_curve, idx_ptr, t_sec):
    if not risk_curve:
        return idx_ptr
    while idx_ptr + 1 < len(risk_curve) and risk_curve[idx_ptr + 1][0] <= t_sec:
        idx_ptr += 1
    score = risk_curve[idx_ptr][1]
    bar_w, bar_h = 200, 18
    x0, y0 = w - bar_w - 10, 10
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    fill_color = (0, 255, 0) if score < 0.5 else (0, 0, 255)
    cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * score), y0 + bar_h), fill_color, -1)
    cv2.putText(frame, f"risk {score:.2f}", (x0, y0 + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return idx_ptr


def build_panel(events, risk_curve, duration, width):
    """Статичная часть панели под кадром: строка на каждый класс из events
    плюс строка риска. Бегунок дорисовывается на каждом кадре."""
    classes = sorted({e[2] for e in events})
    rows = classes + (["risk"] if risk_curve else [])
    h = PANEL_ROW_H * max(len(rows), 1) + 8
    panel = np.full((h, width, 3), 24, np.uint8)
    x_of = lambda t: PANEL_LABEL_W + int((width - PANEL_LABEL_W - 8) * min(max(t / duration, 0), 1))
    for k, name in enumerate(rows):
        y = 4 + k * PANEL_ROW_H
        cv2.putText(panel, name, (6, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (PANEL_LABEL_W, y + 2), (width - 8, y + PANEL_ROW_H - 3), (45, 45, 45), -1)
        if name == "risk":
            y_th = y + PANEL_ROW_H - 3 - int(0.5 * (PANEL_ROW_H - 5))
            cv2.line(panel, (PANEL_LABEL_W, y_th), (width - 8, y_th), (90, 90, 90), 1)
            pts = np.array([[x_of(t), y + PANEL_ROW_H - 3 - int(sc * (PANEL_ROW_H - 5))]
                            for t, sc in risk_curve[::5]], np.int32)
            cv2.polylines(panel, [pts], False, (0, 200, 255), 1, cv2.LINE_AA)
            continue
        for s0, e0, lbl in events:
            if lbl == name:
                cv2.rectangle(panel, (x_of(s0), y + 3), (max(x_of(e0), x_of(s0) + 2), y + PANEL_ROW_H - 4),
                              EVENT_COLORS.get(lbl, (200, 200, 200)), -1)
    return panel, x_of


def draw_risk_pair(frame, scale, explain, score):
    if explain is None or score < 0.3:
        return
    (xa, ya), (xb, yb) = explain["pos"]
    y0 = explain["y0"]
    a = (int(xa * scale), int((ya + y0) * scale))
    b = (int(xb * scale), int((yb + y0) * scale))
    col = (0, 0, 255) if score >= 0.5 else (0, 200, 255)
    cv2.line(frame, a, b, col, 3, cv2.LINE_AA)
    for pnt in (a, b):
        cv2.circle(frame, pnt, 6, col, -1)
    cv2.putText(frame, f"risk {score:.2f}  t*={explain['t_star']}s  a_req={explain['a_req']}",
                (min(a[0], b[0]), max(12, min(a[1], b[1]) - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, col, 2, cv2.LINE_AA)


class VideoSink:
    """Запись кадров BGR: H.264 через ffmpeg из imageio-ffmpeg, иначе mp4v."""

    def __init__(self, path, fps, size):
        self.path, (w, h) = Path(path), size
        self.proc = self.cv = None
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE)
        except (ImportError, OSError):
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame):
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def render(video_path, out_path, zones, records, light_samples, events, risk_curve,
           debug_events, stride, explains=None, max_width=1280, start=0.0, end=None,
           progress=None):
    """Пишет видео с разметкой.

    zones, records — в пикселях этого видео; events — [s, e, label] для
    таймлайна; debug_events — [s, e, label, object_id] (подсветка боксов,
    compute_events_debug); explains — [(t, score, пара риска)] из RiskScorer;
    stride — писать каждый stride-й кадр (как у трекера, чтобы у всех кадров
    были боксы). progress(frac) — необязательный колбэк."""
    zones_np = {n: np.asarray(z, np.float32) for n, z in zones.items()}
    zones_list = {n: z.tolist() for n, z in zones_np.items()}
    light = LightState(light_samples) if light_samples is not None else LightState([])
    events_sorted = sorted(events, key=lambda e: e[0])
    by_frame = {}
    for r in records:
        by_frame.setdefault(r["frame"], []).append(r)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps if fps else 1.0
    scale = min(max_width / w, 1.0)
    out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2    # H.264: чётные размеры
    panel, x_of = build_panel(events_sorted, risk_curve, duration, out_w)
    panel_h = panel.shape[0] // 2 * 2
    panel = panel[:panel_h]
    sink = VideoSink(out_path, fps / stride, (out_w, out_h + panel_h))

    frame_idx, risk_ptr, exp_ptr = -1, 0, 0
    try:
        while cap.grab():
            frame_idx += 1
            if frame_idx % stride:
                continue
            t_sec = frame_idx / fps
            if t_sec < start:
                continue
            if end is not None and t_sec > end:
                break
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            draw_zones(frame, zones_list, scale)

            active_violators = {}
            for s, e, lbl, oid in debug_events:
                if oid is not None and s <= t_sec <= e:
                    active_violators.setdefault(oid, []).append(lbl)
            active_violators = {oid: sorted(set(lbls), key=VIOLATION_PRIORITY.index)
                                for oid, lbls in active_violators.items()}
            recs_now = by_frame.get(frame_idx, [])
            if congestion_active_now(t_sec, debug_events):   # затор — про кластер, подсвечиваем очередь
                for r in recs_now:
                    sid = r.get("stitched_id", r["track_id"])
                    if sid not in active_violators and in_queue_zone(r, zones_np):
                        active_violators[sid] = ["congestion"]
            draw_boxes(frame, recs_now, scale, active_violators)
            draw_light(frame, zones_list, scale, light.at(t_sec))
            active = [lbl for s, e, lbl in events_sorted if s <= t_sec <= e]
            draw_events(frame, out_w, out_h, active, [ev for ev in events_sorted if ev[0] <= t_sec][-6:])
            risk_ptr = draw_risk(frame, out_w, risk_curve, risk_ptr, t_sec)
            if explains:
                while exp_ptr + 1 < len(explains) and explains[exp_ptr + 1][0] <= t_sec:
                    exp_ptr += 1
                t_e, sc_e, ex = explains[exp_ptr]
                if abs(t_e - t_sec) < 0.2:
                    draw_risk_pair(frame, scale, ex, sc_e)
            cv2.putText(frame, f"t={t_sec:6.2f}s", (out_w - 120, out_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            pnl = panel.copy()
            cv2.line(pnl, (x_of(t_sec), 0), (x_of(t_sec), pnl.shape[0]), (255, 255, 255), 1)
            sink.write(np.vstack([frame, pnl]))
            if progress is not None and frame_idx % (stride * 30) == 0 and n_frames:
                progress(frame_idx / n_frames)
    finally:
        cap.release()
        sink.close()
    return Path(out_path)

