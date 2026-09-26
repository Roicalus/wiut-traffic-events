"""Tracking vehicles/people in camera video: YOLO + ByteTrack.

run_tracker() is a reusable function with no file I/O: it is called by
pipeline.extract() (from solution.detect_events()). The CLI (main()) is only for local debugging
of a single video (writes JSON and, optionally, a preview with boxes to disk).

COCO classes: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
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
# both paths are absolute (resolved from this file's location, not from cwd) —
# so run_submission.py finds the weights and the custom tracker regardless of
# which directory it was launched from


_DEVICE = None
READ_AHEAD = 4          # frames in the reader queue (4K BGR ~ 25 MB each)
WARMUP_IMGSZ = 1280     # as in run_tracker: warm up at the same input size


# The top of the frame is not detected: only the strip above the topmost scene zone
# (the roadway starts at y = 91 of 2160 in the reference frame). It used to be
# 18 %, and the crop cut off the far lanes, the stop and 69 % of the zone in front of
# the queue. Part B uses the same constant.
CROP_TOP_FRAC = 0.04


def _frame_reader(cap, stride_box, q, stop):
    """Reader thread: decodes the video and puts every
    stride_box[0]-th frame into the queue. Decoding 4K H.264 on the CPU is the main
    time cost, and in a separate thread it runs in parallel with the detector on the GPU
    (cv2 and torch release the GIL). Frame order is unchanged, the result does not
    depend on thread speed."""
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
    """0 (the first CUDA card) if it ACTUALLY computes, otherwise "cpu".

    torch.cuda.is_available() can be True even though the torch build has no kernels
    for this card (e.g. RTX 50xx / sm_120 with a cu121 build): the failure then
    happens only on the first model call. So we try to run a
    tiny operation on the GPU and fall back to the CPU on error.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    forced = os.environ.get("WIUT_DEVICE")      # demo on ZeroGPU: CUDA is not there yet at import time
    if forced:
        _DEVICE = int(forced) if forced.isdigit() else forced
        return _DEVICE
    try:
        import torch
    except ImportError:
        _DEVICE = "cpu"
        return _DEVICE
    _DEVICE = "cpu"
    # is_available() can be True with device_count() == 0 (CUDA_VISIBLE_DEVICES="",
    # broken driver) — then ultralytics fails on device=0
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            float((torch.ones(8, device="cuda") * 2).sum().item())
            _DEVICE = 0
        except Exception as exc:
            print("WARNING: CUDA is present but does not work with this torch build "
                  f"({str(exc).splitlines()[0]}). Running on CPU. For RTX 50xx install torch "
                  "with cu128: pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu128")
    return _DEVICE


def set_device(device) -> None:
    """Switch the device for all models ("cpu" or a GPU index). Needed by
    the ZeroGPU demo: the GPU exists only inside an @spaces.GPU call.
    ultralytics recreates the predictor itself when the device changes."""
    global _DEVICE
    _DEVICE = device


def run_tracker(video_path, model=None, imgsz=1280, stride=3, conf=0.1,
                 tracker=DEFAULT_TRACKER, crop_top_frac=None, device=None,
                 on_frame=None, time_budget_sec=None, max_stride=6, half=None):
    """Runs YOLO+ByteTrack over a video and returns (meta, records) in memory.

    Args:
        video_path: path to the .mp4.
        model: an already loaded ultralytics.YOLO (reuse it across
            videos — loading weights is not free); if None,
            DEFAULT_MODEL is loaded once inside the call.
        on_frame: optional callback(frame_bgr_cropped, t_sec, result) —
            called on every processed frame. For dev purposes only
            (light_state/obstacle_fire/preview); RiskEstimator must NOT reuse
            this pass — it needs its own causal pass over the
            frames, see src/risk.py.

        time_budget_sec: if set, acts as a safety guard: every 50
            processed frames we project the time of the whole pass, and if it
            exceeds the budget, increase stride (up to max_stride). Slightly
            sparser tracking is better than a video scored as empty.
        half: fp16 inference (default: yes if CUDA is available; on a T4 this is
            ~1.5-2x YOLO speed).
        conf: detector threshold. Deliberately low: ByteTrack itself splits
            detections into confident ones (>= track_high_thresh, new tracks only
            from new_track_thresh=0.6) and weak ones (>= track_low_thresh=0.1),
            which only extend existing tracks via overlap.
            With conf=0.35 the weak ones never reached the tracker: a courier on a moped
            (C3896, 20-27 s) was split into 3 tracks with a 2 s gap; with 0.1 the median
            track length went 7.5 -> 9.7 s, moped coverage 43 -> 61 of 80 frames.

    Returns:
        meta: dict with video/fps/width/height/n_frames_total/stride/...
        records: list of dict with fields frame, t_sec, track_id, cls,
            cls_name, conf, x1, y1, x2, y2 (coordinates in the original
            video resolution).
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

    crop_top_frac = CROP_TOP_FRAC if crop_top_frac is None else crop_top_frac
    y_offset = int(height * crop_top_frac)

    records = []
    cap = cv2.VideoCapture(str(video_path))
    stride_box = [stride]          # the reader looks here: the safety guard may raise stride
    frames_q: queue.Queue = queue.Queue(maxsize=READ_AHEAD)
    stop = threading.Event()
    reader = threading.Thread(target=_frame_reader, args=(cap, stride_box, frames_q, stop), daemon=True)
    reader.start()
    frame_idx = -1
    n_processed = 0
    first_call = True
    stride_changes = []
    t_start = time.perf_counter()
    # The first frames on the GPU are slow (CUDA initialisation, cuDNN kernel selection):
    # they inflate the projection 2-3x. Speed is measured only after
    # WARMUP_FRAMES processed frames, and decisions are made no earlier than
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
            # persist=False on the FIRST frame of a video recreates the trackers: the model
            # is cached across videos, and with persist=True always, the
            # ByteTrack state (active tracks, id counter) would leak from the
            # previous clip into the next one.
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
                rate = (now - t_warm) / max(frame_idx - f_warm, 1)   # seconds per video frame after warm-up
                projected = (now - t_start) + rate * (n_frames_total - frame_idx - 1)
                if projected > time_budget_sec:
                    stride += 1
                    stride_box[0] = stride
                    stride_changes.append((round(t_sec, 1), stride))
                    print(f"[track] projection {projected:.0f}s > budget {time_budget_sec:.0f}s "
                          f"-> stride={stride} from t={t_sec:.0f}s", flush=True)

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
        stop.set()                  # the reader may still be waiting for room in the queue
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
    ap.add_argument("--crop-top-frac", type=float, default=CROP_TOP_FRAC)
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
    print(f"Done: {len(records)} records in {elapsed:.0f}s, max track_id={max_id}")
    print(f"JSON: {out_path}")
    if writer is not None:
        print(f"Preview: {out_path.with_suffix('.preview.mp4')}")


if __name__ == "__main__":
    main()
