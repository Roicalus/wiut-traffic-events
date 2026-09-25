"""light_state.py — классифицирует цвет сигнала светофора в ROI 'light_roi'
из zones.json на каждом N-м кадре видео.

В сабмите светофор читается внутри прохода трекера (LightScanner,
pipeline.extract): отдельного декодирования видео нет. run_light_state() и
CLI (main()) — только для отладки: отдельный проход, JSON и распределение
состояний, чтобы проверить, что ROI не промахнулся мимо светофора.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MIN_BRIGHT_PIXELS = 15  # меньше этого ярких пикселей нужного цвета -> "unknown",
                          # не гадаем на шуме в маленьком ROI


def classify_roi_hue(roi_bgr, min_pixels=MIN_BRIGHT_PIXELS) -> str:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    red_mask = (cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                | cv2.inRange(hsv, (170, 100, 100), (180, 255, 255)))
    yellow_mask = cv2.inRange(hsv, (18, 100, 100), (35, 255, 255))
    green_mask = cv2.inRange(hsv, (45, 80, 80), (90, 255, 255))
    counts = {
        "red": int((red_mask > 0).sum()),
        "yellow": int((yellow_mask > 0).sum()),
        "green": int((green_mask > 0).sum()),
    }
    best = max(counts, key=counts.get)
    if counts[best] < min_pixels:
        return "unknown"
    return best


# Режим классификации:
#   "position" — какая из трёх секций (верх/середина/низ) горит. Не зависит
#                от оттенка освещения (закат, фары, стоп-сигналы за рамкой).
#                Требует, чтобы light_roi плотно облегал ВЕРТИКАЛЬНЫЙ корпус
#                светофора (сверху красный, снизу зелёный).
#   "hue"      — старый способ: считать пиксели нужного цвета в рамке.
#   "auto"     — position, если рамка вытянута по вертикали (h/w >= 1.5).
LIGHT_MODE = "auto"
# Днём на солнце горящая лампа тусклая: 98-й перцентиль V горящей секции
# 45-115 при 0-40 у погашенных (C3896, C3897); на закате/в сумерках — 255.
# Поэтому решает РАЗНИЦА с соседней секцией, а абсолютный порог — только
# отсечка "ничего не горит". Было 150: дневные ролики целиком "unknown".
# С этими порогами на всех 4 сэмплах читается один и тот же цикл:
# красный ~37 с -> зелёный ~37 с -> жёлтый 3 с.
LAMP_MIN_V = 40          # яркость горящей лампы (0-255), ниже — не горит
LAMP_MIN_S = 70          # горящая лампа насыщенная, серый корпус/асфальт — нет
LAMP_MARGIN = 35         # на сколько горящая секция ярче остальных
INNER_X, INNER_Y = 0.2, 0.04   # отрезаем края рамки: туда попадает фон (небо, листва)
OCCLUDE_FRAC = 0.15      # доля рамки светофора под боксом машины = перекрытие
MIN_CONFIDENT_FRAC = 0.3  # реже уверенных показаний — рамка не на светофоре, не доверяем
OCCLUDER_BELOW_PX = 60   # низ бокса перекрывающей машины ниже низа рамки (она ближе)


def _lamp_scores(roi_bgr):
    h0, w0 = roi_bgr.shape[:2]
    dx, dy = int(w0 * INNER_X), int(h0 * INNER_Y)
    if w0 - 2 * dx >= 4 and h0 - 2 * dy >= 6:
        roi_bgr = roi_bgr[dy:h0 - dy, dx:w0 - dx]
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi_bgr, (3, 3), 0), cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    v[hsv[..., 1] < LAMP_MIN_S] = 0          # только насыщенные (цветные) пиксели
    h = v.shape[0]
    bands = (v[: h // 3], v[h // 3: 2 * h // 3], v[2 * h // 3:])
    return [float(np.percentile(b, 98)) if b.size else 0.0 for b in bands]


def classify_roi_position(roi_bgr):
    scores = _lamp_scores(roi_bgr)
    order = np.argsort(scores)[::-1]
    best, second = scores[order[0]], scores[order[1]]
    if best < LAMP_MIN_V or best - second < LAMP_MARGIN:
        return "unknown"
    return ("red", "yellow", "green")[int(order[0])]


def classify_roi(roi_bgr, min_pixels=MIN_BRIGHT_PIXELS, mode=None) -> str:
    mode = mode or LIGHT_MODE
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"
    if mode == "auto":
        h, w = roi_bgr.shape[:2]
        mode = "position" if h >= 1.5 * w else "hue"
    if mode == "position":
        return classify_roi_position(roi_bgr)
    return classify_roi_hue(roi_bgr, min_pixels)


def bbox_from_zone(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


SKIP_YELLOW_CONFIRM_SEC = 3.0   # зелёный -> красный без жёлтого: только если держится столько
HOLD_MAX_SEC = 45.0             # дольше фазы без уверенного показания — цвет неизвестен


def smooth_states(raw, confirm_frames=2, skip_yellow_sec=SKIP_YELLOW_CONFIRM_SEC,
                  hold_max_sec=HOLD_MAX_SEC):
    """Sticky-hold сглаживание: [(t, state)] -> [[t, state]].

    Последний УВЕРЕННЫЙ (не "unknown") цвет держится, пока не встретится
    ДРУГОЙ уверенный цвет подряд confirm_frames раз. "unknown" (блик,
    перекрытие ROI, тёмный кадр) не сбрасывает состояние.

    Цикл светофора — зелёный -> жёлтый (3 с) -> красный. Прямой скачок
    зелёный -> красный почти всегда артефакт: машина закрыла зелёную
    секцию, и тусклая негорящая красная линза днём читается как горящая
    (C3896, 189.8-192.5 с — ложный red_light). Такой переход принимается,
    только если красный держится skip_yellow_sec (жёлтый иногда
    пропускается при 10 Гц — тогда красный просто опоздает на эти секунды).

    Без уверенных показаний дольше hold_max_sec (длиннее одной фазы: ROI
    закрыт, блики, камера уехала) цвет сбрасывается в "unknown", иначе одно
    старое "красный" держалось бы минутами и давало ложные red_light.
    """
    result = []
    last_known = "unknown"
    candidate, candidate_count, candidate_t0 = None, 0, None
    last_seen = None
    for t, state in raw:
        if state == "unknown":
            if last_seen is not None and t - last_seen > hold_max_sec:
                last_known = "unknown"
        elif state == last_known:
            last_seen = t
            candidate, candidate_count = None, 0
        else:
            if state == candidate:
                candidate_count += 1
            else:
                candidate, candidate_count, candidate_t0 = state, 1, t
            skips_yellow = last_known == "green" and candidate == "red"
            if candidate_count >= confirm_frames and (
                    not skips_yellow or t - candidate_t0 >= skip_yellow_sec):
                last_known, last_seen = candidate, t
                candidate, candidate_count = None, 0
        result.append([round(t, 3), last_known])
    return result


class LightScanner:
    """Потоковый классификатор светофора: кормится кадрами из прохода
    трекера (track.run_tracker(on_frame=...)), отдельного декодирования
    4K-видео не нужно — это экономит целый проход по ролику.

    Кадр может быть обрезан сверху: передайте full_height, смещение
    посчитается само (как в ObstacleFireScanner).
    """

    def __init__(self, zones, full_height=None, min_pixels=MIN_BRIGHT_PIXELS):
        if "light_roi" not in zones:
            raise ValueError("В zones нет 'light_roi'")
        self.box = bbox_from_zone(zones["light_roi"])
        self.n_occluded = 0
        self.full_height = full_height
        self.min_pixels = min_pixels
        self.raw = []
        self.failed = False

    def feed(self, frame, t_sec):
        y_off = (self.full_height - frame.shape[0]) if self.full_height else 0
        x1, y1, x2, y2 = self.box
        roi = frame[max(0, y1 - y_off):max(0, y2 - y_off), max(0, x1):max(0, x2)]
        state = classify_roi(roi, self.min_pixels) if roi.size else "unknown"
        self.raw.append((t_sec, state))

    def occluded(self, result, y_off) -> bool:
        """ROI закрыт машиной, которая стоит БЛИЖЕ к камере, чем светофор:
        бокс накрывает >= OCCLUDE_FRAC рамки, а его низ (точка на земле)
        ниже низа рамки больше чем на OCCLUDER_BELOW_PX. Машины на дальних
        полосах за светофором тоже пересекают рамку на картинке, но стоят
        выше по кадру и ничего не закрывают."""
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return False
        x1, y1, x2, y2 = self.box
        area = max((x2 - x1) * (y2 - y1), 1)
        for bx1, by1, bx2, by2 in result.boxes.xyxy.cpu().numpy():
            by1, by2 = by1 + y_off, by2 + y_off
            inter = max(0.0, min(x2, bx2) - max(x1, bx1)) * max(0.0, min(y2, by2) - max(y1, by1))
            if inter / area >= OCCLUDE_FRAC and by2 > y2 + OCCLUDER_BELOW_PX:
                return True
        return False

    def on_tracker_frame(self, cropped_frame, t_sec, result=None):
        if self.failed:
            return
        try:
            y_off = (self.full_height - cropped_frame.shape[0]) if self.full_height else 0
            if self.occluded(result, y_off):
                self.n_occluded += 1
                self.raw.append((t_sec, "unknown"))
                return
            self.feed(cropped_frame, t_sec)
        except Exception as exc:
            self.failed = True
            print(f"[light_state] упал ({exc}); red_light/stop_line пропущены")

    def samples(self, confirm_frames=2):
        """Сглаженный ряд состояний, или None, если светофору нельзя верить:
        сканер упал или уверенных показаний меньше MIN_CONFIDENT_FRAC (на
        сэмплах — 91-100%; мало — значит, рамка не на светофоре: совмещение
        ошиблось или светофор заменили). Тогда red_light/stop_line для видео
        не выдаются, а congestion считается по длительности — лучше, чем
        события по мусорному "цвету"."""
        if self.failed or not self.raw:
            return None
        seen = [s for _, s in self.raw if s != "unknown"]
        frac = len(seen) / max(len(self.raw) - self.n_occluded, 1)
        if frac < MIN_CONFIDENT_FRAC:
            print(f"[light_state] светофор читается уверенно только в {frac:.0%} кадров — "
                  f"не используется (red_light/stop_line для этого видео не выдаются)")
            return None
        return smooth_states(self.raw, confirm_frames)


def run_light_state(video_path, zones, stride=3, min_pixels=MIN_BRIGHT_PIXELS,
                     confirm_frames=2):
    """Отдельный проход по видео (только для CLI/отладки; в pipeline.extract
    светофор читается внутри прохода трекера через LightScanner).
    Возвращает список [t_sec, "red"|"yellow"|"green"|"unknown"]."""
    scanner = LightScanner(zones, min_pixels=min_pixels)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = -1
    while cap.grab():
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        scanner.feed(frame, frame_idx / fps)
    cap.release()
    return scanner.samples(confirm_frames)


# ---------------------------------------------------------------- CLI (dev only)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--confirm-frames", type=int, default=2,
                     help="сколько подряд уверенных (не unknown) сэмплов нового цвета "
                          "нужно для переключения состояния (1 = сразу по первому же)")
    ap.add_argument("--min-pixels", type=int, default=MIN_BRIGHT_PIXELS)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path("src/light") / (Path(args.video).stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zones = json.loads(Path(args.zones).read_text())
    smoothed = run_light_state(args.video, zones, stride=args.stride,
                                confirm_frames=args.confirm_frames, min_pixels=args.min_pixels)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"video": Path(args.video).name, "fps": fps, "samples": smoothed}, f, indent=1)

    counts = {}
    for _, s in smoothed:
        counts[s] = counts.get(s, 0) + 1
    print(f"{len(smoothed)} сэмплов -> {out_path}")
    print("Распределение состояний:", counts)
    if counts.get("unknown", 0) == len(smoothed):
        print("ВНИМАНИЕ: весь ролик 'unknown' — светофор ни разу не был уверенно "
              "распознан, проверь ROI (light_roi) и/или --min-pixels")
    elif counts.get("red", 0) == 0:
        print("ВНИМАНИЕ: 'red' ни разу не встретился — проверь ROI (light_roi) и/или --min-pixels")


if __name__ == "__main__":
    main()
