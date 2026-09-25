"""Трекинг машин/людей на видео с камеры: YOLO + ByteTrack.

run_tracker() — переиспользуемая функция без файлового I/O: её вызывает
pipeline.extract() (из solution.detect_events()). CLI (main()) — только для локальной отладки
одного видео (пишет JSON и, опционально, превью с боксами на диск).

Классы COCO: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
"""
import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import cv2

CLASSES_OF_INTEREST = [0, 1, 2, 3, 5, 7]
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DEFAULT_MODEL = str(Path(__file__).resolve().parent.parent / "weights" / "yolo11s.pt")
DEFAULT_TRACKER = str(Path(__file__).resolve().parent / "bytetrack_long.yaml")
# оба пути абсолютные (считаются от расположения этого файла, не от cwd) —
# так run_submission.py находит веса и кастомный трекер независимо от
# того, из какой директории его запустили


_DEVICE = None
READ_AHEAD = 4          # кадров в очереди читателя (4K BGR ~ 25 МБ каждый)
WARMUP_IMGSZ = 1280     # как в run_tracker: прогрев на том же размере входа


def _frame_reader(cap, stride_box, q, stop):
    """Поток-читатель: декодирует видео и кладёт в очередь каждый
    stride_box[0]-й кадр. Декодирование 4K H.264 на CPU — главный расход
    времени, и в отдельном потоке оно идёт параллельно с детектором на GPU
    (cv2 и torch отпускают GIL). Порядок кадров тот же, результат не
    зависит от скорости потоков."""
    idx = -1
    try:
        while not stop.is_set() and cap.grab():
            idx += 1
            if idx % stride_box[0]:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            while not stop.is_set():
                try:
                    q.put((idx, frame), timeout=0.5)
                    break
                except queue.Full:
                    pass
    finally:
        q.put(None)


def _pick_device():
    """0 (первая CUDA-карта), если она РЕАЛЬНО считает, иначе "cpu".

    torch.cuda.is_available() бывает True, хотя ядра под эту карту в сборке
    torch нет (например, RTX 50xx / sm_120 со сборкой cu121): тогда падение
    случается только при первом вызове модели. Поэтому пробуем выполнить
    крошечную операцию на GPU и при ошибке откатываемся на CPU.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    forced = os.environ.get("WIUT_DEVICE")      # демо на ZeroGPU: при импорте CUDA ещё нет
    if forced:
        _DEVICE = int(forced) if forced.isdigit() else forced
        return _DEVICE
    try:
        import torch
    except ImportError:
        _DEVICE = "cpu"
        return _DEVICE
    _DEVICE = "cpu"
    # is_available() бывает True при device_count() == 0 (CUDA_VISIBLE_DEVICES="",
    # сломанный драйвер) — тогда ultralytics падает на device=0
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            float((torch.ones(8, device="cuda") * 2).sum().item())
            _DEVICE = 0
        except Exception as exc:
            print("WARNING: CUDA есть, но не работает с этой сборкой torch "
                  f"({str(exc).splitlines()[0]}). Работаю на CPU. Для RTX 50xx поставьте torch "
                  "с cu128: pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu128")
    return _DEVICE


def set_device(device) -> None:
    """Переключить устройство для всех моделей ("cpu" или номер GPU). Нужно
    демо на ZeroGPU: видеокарта есть только внутри @spaces.GPU-вызова.
    ultralytics сам пересоздаёт предиктор, когда меняется device."""
    global _DEVICE
    _DEVICE = device


def run_tracker(video_path, model=None, imgsz=1280, stride=3, conf=0.1,
                 tracker=DEFAULT_TRACKER, crop_top_frac=0.18, device=None,
                 on_frame=None, time_budget_sec=None, max_stride=6, half=None):
    """Гоняет YOLO+ByteTrack по видео и возвращает (meta, records) в памяти.

    Args:
        video_path: путь к .mp4.
        model: уже загруженный ultralytics.YOLO (переиспользуйте между
            видео — загрузка весов не бесплатна); если None, грузится
            DEFAULT_MODEL один раз внутри вызова.
        on_frame: опциональный callback(frame_bgr_cropped, t_sec, result) —
            вызывается на каждом обработанном кадре. Только для дев-нужд
            (light_state/obstacle_fire/превью); RiskEstimator НЕ должен переиспользовать
            этот проход — ему нужен собственный каузальный проход по
            кадрам, см. src/risk.py.

        time_budget_sec: если задано — предохранитель: каждые 50
            обработанных кадров проецируем время всего прохода, и если оно
            вылезает за бюджет, увеличиваем stride (до max_stride). Лучше
            чуть более редкий трекинг, чем видео, зачтённое пустым.
        half: fp16-инференс (по умолчанию — да, если есть CUDA; на T4 это
            ~1.5-2x к скорости YOLO).
        conf: порог детектора. Низкий намеренно: ByteTrack сам делит
            детекции на уверенные (>= track_high_thresh, новые треки — только
            от new_track_thresh=0.6) и слабые (>= track_low_thresh=0.1),
            которыми лишь продлевает существующие треки через перекрытия.
            При conf=0.35 слабые до трекера не доходили: курьер на мопеде
            (C3896, 20-27 с) рвался на 3 трека с дырой 2 с; с 0.1 — медиана
            длины трека 7.5 -> 9.7 с, покрытие мопеда 43 -> 61 из 80 кадров.

    Returns:
        meta: dict с video/fps/width/height/n_frames_total/stride/...
        records: list of dict с полями frame, t_sec, track_id, cls,
            cls_name, conf, x1, y1, x2, y2 (координаты в исходном
            разрешении видео).
    """
    video_path = Path(video_path)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(DEFAULT_MODEL)
    if device is None:
        device = _pick_device()
    if half is None:
        half = device != "cpu"

    cap_meta = cv2.VideoCapture(str(video_path))
    fps = cap_meta.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_total = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_meta.release()

    y_offset = int(height * crop_top_frac)

    records = []
    cap = cv2.VideoCapture(str(video_path))
    stride_box = [stride]          # читатель смотрит сюда: предохранитель может поднять stride
    frames_q: queue.Queue = queue.Queue(maxsize=READ_AHEAD)
    stop = threading.Event()
    reader = threading.Thread(target=_frame_reader, args=(cap, stride_box, frames_q, stop), daemon=True)
    reader.start()
    frame_idx = -1
    n_processed = 0
    first_call = True
    stride_changes = []
    t_start = time.perf_counter()
    # Первые кадры на GPU медленные (инициализация CUDA, подбор ядер cuDNN):
    # по ним прогноз завышается в 2-3 раза. Скорость меряем только после
    # WARMUP_FRAMES обработанных кадров, а решения принимаем не раньше
    # GUARD_MIN_FRAMES.
    WARMUP_FRAMES, GUARD_MIN_FRAMES = 30, 150
    t_warm, f_warm = None, None

    try:
        while True:
            item = frames_q.get()
            if item is None:
                break
            frame_idx, frame = item

            cropped = frame[y_offset:, :]
            # persist=False на ПЕРВОМ кадре видео пересоздаёт трекеры: модель
            # кэшируется между видео, и с persist=True всегда состояние
            # ByteTrack (активные треки, счётчик id) протекало бы из
            # предыдущего ролика в следующий.
            r = model.track(
                cropped,
                persist=not first_call,
                tracker=tracker,
                classes=CLASSES_OF_INTEREST,
                imgsz=imgsz,
                conf=conf,
                device=device,
                half=half,
                verbose=False,
            )[0]
            first_call = False
            n_processed += 1

            t_sec = frame_idx / fps

            if n_processed == WARMUP_FRAMES:
                t_warm, f_warm = time.perf_counter(), frame_idx
            if (time_budget_sec and t_warm is not None and n_processed >= GUARD_MIN_FRAMES
                    and n_processed % 50 == 0 and stride < max_stride and n_frames_total):
                now = time.perf_counter()
                rate = (now - t_warm) / max(frame_idx - f_warm, 1)   # сек на кадр видео после разгона
                projected = (now - t_start) + rate * (n_frames_total - frame_idx - 1)
                if projected > time_budget_sec:
                    stride += 1
                    stride_box[0] = stride
                    stride_changes.append((round(t_sec, 1), stride))
                    print(f"[track] прогноз {projected:.0f}s > бюджета {time_budget_sec:.0f}s "
                          f"-> stride={stride} с t={t_sec:.0f}s", flush=True)

            if r.boxes is not None and r.boxes.id is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                clss = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), tid, cls, conf_ in zip(xyxy, ids, clss, confs):
                    records.append({
                        "frame": int(frame_idx),
                        "t_sec": round(float(t_sec), 3),
                        "track_id": int(tid),
                        "cls": int(cls),
                        "cls_name": COCO_NAMES.get(int(cls), str(cls)),
                        "conf": round(float(conf_), 3),
                        "x1": round(float(x1), 1), "y1": round(float(y1 + y_offset), 1),
                        "x2": round(float(x2), 1), "y2": round(float(y2 + y_offset), 1),
                    })

            if on_frame is not None:
                on_frame(cropped, t_sec, r)
    finally:
        stop.set()                  # читатель мог остаться ждать места в очереди
        while reader.is_alive():
            try:
                frames_q.get(timeout=0.1)
            except queue.Empty:
                pass
        cap.release()

    meta = {
        "video": video_path.name,
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames_total": n_frames_total,
        "stride": stride,
        "stride_changes": stride_changes,
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
        "imgsz": imgsz,
        "model": getattr(model, "ckpt_path", DEFAULT_MODEL),
        "tracker": tracker,
        "conf": conf,
        "crop_top_frac": crop_top_frac,
    }
    return meta, records


# ---------------------------------------------------------------- CLI (dev only)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--tracker", default=DEFAULT_TRACKER)
    ap.add_argument("--crop-top-frac", type=float, default=0.18)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--preview-width", type=int, default=1280)
    args = ap.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else Path("src/tracks") / f"{video_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.model)

    writer = None
    if args.save_video:
        cap = cv2.VideoCapture(str(video_path))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        preview_h = int(args.preview_width * h / w)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path.with_suffix(".preview.mp4")), fourcc,
                                  fps / args.stride, (args.preview_width, preview_h))

    def preview(cropped, t_sec, r):
        writer.write(cv2.resize(r.plot(), (args.preview_width, preview_h)))

    t0 = time.perf_counter()
    meta, records = run_tracker(
        video_path, model=model, imgsz=args.imgsz, stride=args.stride, conf=args.conf,
        tracker=args.tracker, crop_top_frac=args.crop_top_frac,
        on_frame=preview if writer is not None else None,
    )
    elapsed = time.perf_counter() - t0

    if writer is not None:
        writer.release()

    out_path.write_text(json.dumps({"meta": meta, "tracks": records}, indent=1))
    max_id = max((r["track_id"] for r in records), default=0)
    print(f"Готово: {len(records)} записей за {elapsed:.0f}с, максимальный track_id={max_id}")
    print(f"JSON: {out_path}")
    if writer is not None:
        print(f"Превью: {out_path.with_suffix('.preview.mp4')}")


if __name__ == "__main__":
    main()
