"""export_site.py — данные и медиа для сайта команды из артефактов репозитория.

Всё на сайте о сэмплах получено этим скриптом из того же, что видит жюри:
predictions_samples.json (сабмит), кэш прохода Part A (tools/dev_loop.py) и
видео с разметкой (tools/visualize_debug.py). Ничего не рисуется руками.

    python tools/dev_loop.py --videos samples            # кэш, если его нет
    python tools/visualize_debug.py --video samples/X.MP4 --predictions predictions_samples.json   # для каждого
    python tools/export_site.py --site ../site

Пишет в <site>/public:
  data/predictions_samples.json   копия файла сабмита
  data/samples.json               параметры роликов, освещение, смещение камеры, светофор,
                                  примеры классов, неудачные случаи
  data/eda.json                   объекты по классам во времени, плотность потока, рисунки
  media/annotated/*.mp4           размеченные видео (H.264, 960 px, faststart)
  media/examples/*.mp4            клипы-примеры классов
  media/failures/*.jpg            кадры неудачных случаев
  media/heatmaps, trajectories, eda/*.png
  media/hero.mp4, data/hero.json  фрагмент C3896 без разметки и наши треки на нём (главная)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import align  # noqa: E402
from src.pipeline import get_zones, load_obs, reference_view  # noqa: E402
from src.rules import VEHICLE_CLASSES, annotate, group_by_object  # noqa: E402

WEB_WIDTH = 960
POSTER_AT_SEC = 20.0          # кадр-превью размеченного ролика
HERO = ("C3896.MP4", 44.0, 84.0)  # фрагмент для главной: разворот, остановка, проезд на красный
BIN_SEC = 5.0
LIGHTING = {"C3896": "day, direct sun", "C3897": "day, direct sun", "C3902": "sunset", "C3905": "dusk, headlights on"}
SERIES = {"car": [2], "bus": [5], "truck": [7], "motorcycle / bicycle": [1, 3], "person": [0]}

# Примеры классов: реальные события из predictions_samples.json, проверенные глазами в дебаг-видео.
EXAMPLES = [
    ("illegal_turn", "C3896.MP4", 49.6, "A car from the main road drives deep into the junction and makes "
     "a U-shaped turn back onto the lower street: the forbidden route (the allowed right turn is right after "
     "the far crossing)."),
    ("red_light", "C3896.MP4", 79.0, "A vehicle crosses the stop line while the signal for its approach is red."),
    ("stopped_vehicle", "C3896.MP4", 143.9, "A lone car stands on the junction for more than 10 s, not packed "
     "in a queue: stopping on the junction is an event even if the car leaves with the next green phase."),
    ("congestion", "C3896.MP4", 39.0, "The junction jams: a dense block of cars stands for longer than one "
     "signal phase. An ordinary red-light queue is not congestion; it clears on green."),
    ("jaywalking", "C3897.MP4", 142.3, "Pedestrians leave the zebra and cut across the asphalt towards the "
     "near crossing."),
    ("stop_line", "C3905.MP4", 77.7, "At dusk a car stops on red past the stop line, its front on the far "
     "zebra, and waits there for the green instead of behind the line."),
]

# Неудачные и спорные случаи — честно, с объяснением.
FAILURES = [
    ("Close following read as a near conflict", "C3897.MP4", 183.1,
     "Two cars in the far lane, one closing in on the other in slow traffic. The submitted risk curve peaks "
     "at 0.498 here, a hair below the 0.5 alarm line, although nothing dangerous happens. Distances are "
     "measured in image space: on the far side of the junction perspective squeezes a normal following gap "
     "into a few pixels. A ground-plane (bird's-eye) projection with metric gaps is the fix we would try."),
    ("Jaywalking or a desire line?", "C3897.MP4", 150.0,
     "Almost every pedestrian phase, people step off the far zebra and cut diagonally across the asphalt to "
     "the near crossing. By the definition it is jaywalking and we report it; if the annotators treat this "
     "habitual path as normal, these events are false positives."),
    ("A lone stop merged into a whole-video segment", "C3902.MP4", 120.0,
     "A car with hazard lights stands for minutes near the bus stop. Same-class events that overlap must be "
     "merged into one segment, so this stop swallows the shorter stops around it and the video gets one long "
     "stopped_vehicle segment instead of several short ones."),
]


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(src, dst, start=None, duration=None, width=WEB_WIDTH, crf=28):
    cmd = [ffmpeg(), "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.2f}"]
    cmd += ["-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    subprocess.run(cmd, check=True)


def frame_at(video, t):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def write_poster(video, t, dst):
    """Кадр-превью для <video poster> и карточек сайта (WEB_WIDTH, JPEG)."""
    f = frame_at(video, t)
    if f is None:
        return None
    h, w = f.shape[:2]
    f = cv2.resize(f, (WEB_WIDTH, int(h * WEB_WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), f, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return dst


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {"duration": round(n / fps, 2), "fps": round(fps, 2),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return meta


def light_phases(samples):
    """Длительности фаз светофора (с) по сглаженному ряду состояний."""
    phases, cur, t0 = defaultdict(list), None, None
    for t, s in samples or []:
        if s != cur:
            if cur in ("red", "green", "yellow") and t0 is not None:
                phases[cur].append(t - t0)
            cur, t0 = s, t
    # первая и последняя фазы обрезаны началом и концом ролика — не считаем
    return {k: round(float(np.median(v[1:])) if len(v) > 1 else float(v[0]), 1) for k, v in phases.items() if v}


def counts_and_density(obs, duration):
    frames = defaultdict(lambda: defaultdict(int))
    for r in obs.records:
        frames[r["frame"]][r["cls"]] += 1
    fps = obs.meta["fps"]
    bins = defaultdict(lambda: defaultdict(list))
    for f, by_cls in frames.items():
        b = int((f / fps) // BIN_SEC)
        for name, ids in SERIES.items():
            bins[b][name].append(sum(by_cls.get(c, 0) for c in ids))
    n_bins = int(duration // BIN_SEC) + 1
    t = [round(i * BIN_SEC, 1) for i in range(n_bins)]
    series = {name: [round(float(np.mean(bins[i][name])), 1) if bins[i][name] else 0.0 for i in range(n_bins)]
              for name in SERIES}
    first_seen = defaultdict(lambda: 1e9)
    for r in obs.records:
        if r["cls"] in VEHICLE_CLASSES:
            sid = r.get("stitched_id", r["track_id"])
            first_seen[sid] = min(first_seen[sid], r["t_sec"])
    starts = np.array(sorted(first_seen.values()))
    density = [int(((starts >= tt - 30) & (starts < tt + 30)).sum() * 60 / max(min(tt + 30, duration) - max(tt - 30, 0), 1))
               for tt in t]
    return {"t": t, "unit": "objects in frame (mean)", "series": series}, \
           {"t": t, "unit": "vehicles appearing per minute", "values": density}


def heatmap_and_trajectories(obs, base, out_heat, out_traj):
    """Тепловая карта движения (машины — тёплая, пешеходы — голубая) и траектории
    с цветом по направлению движения, на кадре ролика."""
    h, w = base.shape[:2]
    k = WEB_WIDTH / w
    small = cv2.resize(base, (WEB_WIDTH, int(h * k)), interpolation=cv2.INTER_AREA)
    acc = {"veh": np.zeros(small.shape[:2], np.float32), "ped": np.zeros(small.shape[:2], np.float32)}
    traj = (small * 0.45).astype(np.uint8)
    for oid, recs in group_by_object(obs.records).items():
        samples = annotate(recs, {})
        kind = "veh" if recs[0]["cls"] in VEHICLE_CLASSES else "ped" if recs[0]["cls"] == 0 else None
        if kind is None:
            continue
        pts = []
        for s in samples:
            x, y = int(s["g"][0] * k), int(s["g"][1] * k)
            if 0 <= x < WEB_WIDTH and 0 <= y < small.shape[0] and s["speed"] > 0.3:
                acc[kind][y, x] += 1
            pts.append((x, y))
        if kind == "veh" and len(pts) > 20 and samples[-1]["t"] - samples[0]["t"] > 2:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            if np.hypot(x1 - x0, y1 - y0) < 60:
                continue
            hue = int((np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 360) / 2)
            col = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.polylines(traj, [np.int32(pts[::2])], False, col, 1, cv2.LINE_AA)
    out = (small * 0.5).astype(np.float32)
    for kind, cmap in (("veh", cv2.COLORMAP_INFERNO), ("ped", cv2.COLORMAP_OCEAN)):
        a = cv2.GaussianBlur(acc[kind], (0, 0), 6)
        if a.max() > 0:
            a = np.clip(a / np.percentile(a[a > 0], 99), 0, 1)
            col = cv2.applyColorMap((a * 255).astype(np.uint8), cmap).astype(np.float32)
            out = out * (1 - a[..., None] * 0.85) + col * (a[..., None] * 0.85)
    cv2.imwrite(str(out_heat), out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # компас: цвет = направление движения
    cx, cy, r = WEB_WIDTH - 60, 60, 38
    for a in range(0, 360, 10):
        hue = int(a / 2)
        col = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        x, y = int(cx + r * np.cos(np.radians(a))), int(cy + r * np.sin(np.radians(a)))
        cv2.line(traj, (cx, cy), (x, y), col, 3, cv2.LINE_AA)
    cv2.putText(traj, "direction", (cx - 36, cy + r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.imwrite(str(out_traj), traj, [cv2.IMWRITE_JPEG_QUALITY, 88])


def light_cycle_figure(light_by_video, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"red": "#d33", "yellow": "#e8b400", "green": "#2a2", "unknown": "#999"}
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(light_by_video) + 0.8), dpi=110)
    for row, (name, samples) in enumerate(light_by_video.items()):
        for (t0, s), (t1, _) in zip(samples, samples[1:]):
            ax.barh(row, t1 - t0, left=t0, color=colors.get(s, "#999"), height=0.6, linewidth=0)
    ax.set_yticks(range(len(light_by_video)), list(light_by_video))
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Traffic-light state read from the video (signal of the main-road queue)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def alignment_figure(videos, out):
    """Зоны сцены, совмещённые с каждой записью: крупно светофор и переходы."""
    zones = get_zones()
    tiles = []
    for v in videos:
        z, rep = align.aligned_zones(v, zones)
        f = align.first_frame(v, 300)
        for poly in z.values():
            cv2.polylines(f, [poly.astype(np.int32)], len(poly) >= 3, (0, 255, 0), 4)
        crop = cv2.resize(f[450:1450, 1500:3300], (720, 400))
        cv2.putText(crop, f"{v.stem}: shift {rep.get('dx')}x{rep.get('dy')} px, {rep.get('rot_deg')} deg, "
                    f"zoom {rep.get('scale')}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(crop)
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def hero_clip(pub, videos_dir):
    """Главная сайта: фрагмент исходного C3896 без разметки и треки нашего прохода Part A на нём
    (рамки рисует сайт поверх видео, синхронно с его временем). Объекты событий — из
    compute_events_debug, те же, что подсвечивает visualize_debug."""
    import solution
    from src.rules import compute_events_debug
    video, start, end = HERO
    obs = load_obs(ROOT / "cache" / f"{video}.obs.pkl.gz")
    records, zones = reference_view(obs)
    debug = compute_events_debug(records, zones, obs.light_samples, classes=solution.CLASSES)
    w, h = obs.meta["width"], obs.meta["height"]
    frames = defaultdict(list)
    for r in obs.records:
        if start <= r["t_sec"] <= end and r["conf"] >= 0.3:
            frames[round(r["t_sec"] - start, 2)].append(
                [r.get("stitched_id", r["track_id"]), r["cls_name"],
                 round(r["x1"] / w, 4), round(r["y1"] / h, 4), round(r["x2"] / w, 4), round(r["y2"] / h, 4)])
    objects = [[round(s0 - start, 2), round(e0 - start, 2), label, oid] for s0, e0, label, oid in debug
               if oid is not None and e0 > start and s0 < end]
    src = Path(videos_dir) / video
    encode(src, pub / "media" / "hero.mp4", start=start, duration=end - start, width=1280, crf=26)
    write_poster(src, start + 0.5, pub / "media" / "hero.jpg")
    (pub / "data" / "hero.json").write_text(json.dumps({
        "video": video, "start": start, "end": end, "media": "media/hero.mp4", "poster": "media/hero.jpg",
        "frames": sorted([t, boxes] for t, boxes in frames.items()), "event_objects": objects,
    }, separators=(",", ":")), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="папка сайта (public/ внутри)")
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--skip-video", action="store_true", help="не перекодировать видео (только данные)")
    args = ap.parse_args()
    pub = Path(args.site) / "public"
    for d in ("data", "media/annotated", "media/examples", "media/failures", "media/heatmaps",
              "media/trajectories", "media/eda", "media/posters"):
        (pub / d).mkdir(parents=True, exist_ok=True)
    hero_clip(pub, args.videos)
    preds = json.loads((ROOT / "predictions_samples.json").read_text(encoding="utf-8"))
    shutil.copy2(ROOT / "predictions_samples.json", pub / "data" / "predictions_samples.json")
    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() == ".mp4")

    samples, eda = {"videos": {}}, {"findings": [], "counts_over_time": {}, "density": {},
                                   "heatmaps": [], "trajectories": [], "extra": []}
    light_by_video = {}
    for v in videos:
        print(f"[{v.name}]", flush=True)
        obs = load_obs(ROOT / "cache" / f"{v.name}.obs.pkl.gz")
        reference_view(obs)                    # stitched_id в записях, как у правил
        meta = video_meta(v)
        _, rep = align.aligned_zones(v, get_zones())
        phases = light_phases(obs.light_samples)
        light_by_video[v.stem] = obs.light_samples or []
        ev = preds["videos"][v.name]["events"]
        counts = defaultdict(int)
        for _, _, label in ev:
            counts[label] += 1
        notes = (f"Camera vs. the reference view: shift {rep.get('dx')}×{rep.get('dy')} px, rotation "
                 f"{rep.get('rot_deg')}°, zoom {rep.get('scale')} (homography, reference '{rep.get('ref')}'). "
                 f"Signal cycle read from the video: red {phases.get('red', '—')} s, green {phases.get('green', '—')} s, "
                 f"yellow {phases.get('yellow', '—')} s. Events: "
                 + (", ".join(f"{k} {n}" for k, n in sorted(counts.items())) or "none") + ".")
        samples["videos"][v.name] = dict(meta, lighting=LIGHTING.get(v.stem, ""),
                                         annotated_video=f"media/annotated/{v.stem}.mp4",
                                         poster=f"media/posters/{v.stem}.jpg", notes=notes)
        c, d = counts_and_density(obs, meta["duration"])
        eda["counts_over_time"][v.name], eda["density"][v.name] = c, d
        base = frame_at(v, 5.0)
        heatmap_and_trajectories(obs, base, pub / "media" / "heatmaps" / f"{v.stem}.jpg",
                                 pub / "media" / "trajectories" / f"{v.stem}.jpg")
        eda["heatmaps"].append({"video": v.name, "src": f"media/heatmaps/{v.stem}.jpg",
                                "caption": f"{v.name} ({LIGHTING.get(v.stem)}): where moving vehicles (warm) "
                                           "and pedestrians (blue) spend time."})
        eda["trajectories"].append({"video": v.name, "src": f"media/trajectories/{v.stem}.jpg",
                                    "caption": f"{v.name}: vehicle tracks longer than 2 s, colour = direction "
                                               "of travel (compass top right)."})
        if not args.skip_video:
            src = ROOT / "debug" / f"{v.stem}.debug.mp4"
            if src.exists():
                encode(src, pub / "media" / "annotated" / f"{v.stem}.mp4")
                write_poster(src, POSTER_AT_SEC, pub / "media" / "posters" / f"{v.stem}.jpg")
            else:
                print(f"   нет {src} — сначала tools/visualize_debug.py")

    light_cycle_figure(light_by_video, pub / "media" / "eda" / "light_cycle.png")
    alignment_figure(videos, pub / "media" / "eda" / "alignment.jpg")
    eda["extra"] = [
        {"title": "Traffic-light cycle", "src": "media/eda/light_cycle.png",
         "caption": "The signal of the main-road queue, read from the lit section: the same cycle in every "
                    "recording (red ≈37 s, green ≈37 s, yellow 3 s)."},
        {"title": "The camera moves between recordings", "src": "media/eda/alignment.jpg",
         "caption": "Scene zones drawn once on C3896 and mapped to each recording by a homography. Without "
                    "this, the traffic-light box of C3902 missed the signal by ~100 px."},
    ]

    examples = []
    for label, video, start, caption in EXAMPLES:
        match = [e for e in preds["videos"][video]["events"] if e[2] == label and abs(e[0] - start) < 3]
        if not match:
            print(f"   пример {label} {video} {start}: такого события в predictions нет — пропускаю")
            continue
        s, e, _ = match[0]
        clip_start, clip_len = max(0.0, s - 2.0), min(e - s + 4.0, 20.0)
        if not args.skip_video:
            debug = ROOT / "debug" / f"{Path(video).stem}.debug.mp4"
            encode(debug, pub / "media" / "examples" / f"{label}.mp4", start=clip_start, duration=clip_len)
            write_poster(debug, (s + e) / 2 if e - s < 16 else s + 2, pub / "media" / "posters" / f"ex_{label}.jpg")
        examples.append({"label": label, "video": video, "start": s, "end": e,
                         "media": f"media/examples/{label}.mp4", "poster": f"media/posters/ex_{label}.jpg",
                         "caption": caption})
    failures = []
    for i, (title, video, t, text) in enumerate(FAILURES, 1):
        f = frame_at(ROOT / "debug" / f"{Path(video).stem}.debug.mp4", t)
        if f is not None:
            cv2.imwrite(str(pub / "media" / "failures" / f"failure_{i}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
        failures.append({"title": title, "video": video, "start": max(0.0, t - 3), "end": t + 5,
                         "media": f"media/failures/failure_{i}.jpg", "explanation": text})
    samples["class_examples"], samples["failures"] = examples, failures
    eda["findings"] = json.loads((ROOT / "docs" / "eda_findings.json").read_text(encoding="utf-8"))
    (pub / "data" / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    (pub / "data" / "eda.json").write_text(json.dumps(eda, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Готово:", pub)


if __name__ == "__main__":
    main()
