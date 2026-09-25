"""
define_zones.py — кликами по одному кадру задаёт полигоны зон камеры
(зона очереди перед светофором, зоны переходов, проезжая часть, ROI
светофора и т.п.). Результат — zones.json, переиспользуемый для ВСЕХ
видео этой камеры (сэмплы и скрытый тест — тот же ракурс).

Запуск:
    python define_zones.py --video samples/C3896.MP4 --frame 150 --out zones.json

Управление:
    ЛКМ     — добавить точку к текущему полигону
    ПКМ     — замкнуть текущий полигон (нужно >=3 точки; для линии 7
              (solid_line) не нужна — она не замыкается, просто 2 точки)
    1..8    — присвоить только что замкнутому полигону имя из списка ниже
              (после этого можно сразу начинать следующий полигон);
              7 (solid_line) — особый случай: это ЛИНИЯ (ровно 2 точки),
              не полигон, см. src/rules.py detect_solid_line_crossing
    0       — присвоить произвольное имя (единственный случай, когда
              спросит в консоли — используйте только для нестандартных зон)
    U       — отменить последнюю точку текущего полигона
    Z       — удалить последнюю сохранённую (уже названную) зону
    Q       — закончить и сохранить всё в --out

Координаты кликов сохраняются в ИСХОДНОМ разрешении видео, даже если
картинка на экране показана уменьшенной (см. --max-width/--max-height) —
zones.json остаётся валиден для track.py/rules.py, которые работают с
полноразмерными кадрами.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRESET_NAMES = {
    ord('1'): "queue_zone",     # очередь по всем полосам перед переходом — для congestion
    ord('2'): "roadway",        # проезжая часть вне переходов — для jaywalking
    ord('3'): "stop_line",      # линия стоп перед дальним переходом
    ord('4'): "crossing_far",   # переход через магистраль
    ord('5'): "crossing_near",  # диагональный переход по площади
    ord('6'): "light_roi",      # рамка на сигнале светофора — для определения цвета
    ord('7'): "solid_line",     # ЛИНИЯ (ровно 2 точки!) — для solid_line_crossing
    ord('8'): "illegal_turn_exit",  # выезд, куда нельзя попасть с магистрали (см. rules.detect_illegal_turn)
}

# зоны-ЛИНИИ: не полигон, а открытый отрезок — достаточно 2 точек, право-
# кликом замыкать не нужно (см. LINE_MIN_POINTS ниже и src/rules.py,
# detect_solid_line_crossing — там же объяснено, зачем это именно линия,
# а не полигон: cv2.pointPolygonTest не умеет "по какую сторону от прямой").
LINE_PRESETS = {"solid_line"}


def _min_points_for(name: str) -> int:
    if name in LINE_PRESETS or name.startswith("solid_line_"):
        return 2
    return 3


def fit_scale(w, h, max_w, max_h):
    return min(max_w / w, max_h / h, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=150,
                     help="номер кадра для разметки (лучше кадр с трафиком)")
    ap.add_argument("--out", default="zones.json")
    ap.add_argument("--load", action="store_true",
                     help="загрузить существующий --out и дорисовать/заменить зоны (зона с тем же "
                          "именем перезаписывается). Рисуйте на ТОМ ЖЕ видео/кадре, что в zones_ref.json")
    ap.add_argument("--max-width", type=int, default=1600,
                     help="макс. ширина окна на экране (картинка масштабируется под неё)")
    ap.add_argument("--max-height", type=int, default=900,
                     help="макс. высота окна на экране")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= n_frames:
        cap.release()
        raise SystemExit(f"Кадр {args.frame} за пределами видео (всего кадров: {n_frames})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Не удалось прочитать кадр {args.frame} из {args.video}")

    h, w = frame.shape[:2]
    scale = fit_scale(w, h, args.max_width, args.max_height)
    print(f"Кадр {w}x{h}, масштаб показа: {scale:.3f} "
          f"(координаты в zones.json будут в исходном разрешении {w}x{h})")
    disp_w, disp_h = int(w * scale), int(h * scale)

    zones = {}          # name -> polygon в ИСХОДНЫХ координатах видео
    if args.load and Path(args.out).exists():
        zones = json.loads(Path(args.out).read_text())
        print(f"Загружено {len(zones)} зон из {args.out}: {list(zones)}")
        ref_meta = Path(__file__).resolve().parent.parent / "zones_ref.json"
        if ref_meta.exists():
            m = json.loads(ref_meta.read_text())
            if (m.get("video"), m.get("frame")) != (Path(args.video).name, args.frame):
                print(f"ВНИМАНИЕ: опорный кадр зон — {m.get('video')} кадр {m.get('frame')}, а вы "
                      f"рисуете на {Path(args.video).name} кадр {args.frame}. Камера между видео "
                      f"сдвинута — рисуйте на опорном, иначе новые зоны не совпадут со старыми.")
    current = []         # текущий незамкнутый полигон, тоже в исходных координатах
    history = []         # стек действий для отмены: ('point',) | ('zone', name, polygon)
    status = "Кликай ЛКМ точки полигона, ПКМ — замкнуть"

    def undo():
        nonlocal current, status
        if not history:
            status = "Отменять нечего"
            return
        action = history.pop()
        if action[0] == "point":
            if current:
                current.pop()
            status = "Отменена последняя точка"
        elif action[0] == "zone":
            _, name, poly = action
            zones.pop(name, None)
            current = poly  # точки возвращаются в работу — можно перерисовать/переназвать
            status = f"Отменена зона '{name}' — точки снова в работе, назначь имя заново"

    def redraw():
        vis = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        for name, poly in zones.items():
            pts = (np.array(poly, dtype=np.float32) * scale).astype(int)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.putText(vis, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        if current:
            pts = (np.array(current, dtype=np.float32) * scale).astype(int)
            for p in pts:
                cv2.circle(vis, tuple(p), 4, (0, 0, 255), -1)
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, (0, 0, 255), 2)
        legend = " | ".join(f"{k}:{v}" for k, v in
                             {"1": "queue_zone", "2": "roadway", "3": "stop_line",
                              "4": "crossing_far", "5": "crossing_near", "6": "light_roi",
                              "7": "solid_line(2pt)", "8": "illegal_turn_exit"}.items())
        cv2.putText(vis, status, (10, disp_h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 0), 1)
        cv2.putText(vis, legend + "  |  U/Ctrl+Z: отменить", (10, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("define_zones", vis)

    def on_mouse(event, x, y, flags, param):
        nonlocal status
        if event == cv2.EVENT_LBUTTONDOWN:
            # экранные координаты -> исходное разрешение видео
            current.append([x / scale, y / scale])
            history.append(("point",))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current) < 3:
                status = "Нужно минимум 3 точки, чтобы замкнуть полигон (для линии 7 — не нужно, жми 7 сразу после 2 точек)"
            else:
                status = "Полигон замкнут — нажми 1-8 (готовое имя) или 0 (своё)"
            redraw()

    cv2.namedWindow("define_zones", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("define_zones", disp_w, disp_h)
    cv2.setMouseCallback("define_zones", on_mouse)
    redraw()

    while True:
        redraw()
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('u'), 26):  # 'u' или Ctrl+Z
            undo()
        elif key in PRESET_NAMES and len(current) >= _min_points_for(PRESET_NAMES[key]):
            name = PRESET_NAMES[key]
            zones[name] = current
            history.append(("zone", name, current))
            current = []
            status = f"Сохранено: {name} — рисуй следующий полигон"
        elif key == ord('0') and len(current) >= 2:
            cv2.destroyWindow("define_zones")  # чтобы консольный input() точно был в фокусе
            name = input("Своё имя зоны: ").strip()
            cv2.namedWindow("define_zones", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("define_zones", disp_w, disp_h)
            cv2.setMouseCallback("define_zones", on_mouse)
            if name:
                zones[name] = current
                history.append(("zone", name, current))
                status = f"Сохранено: {name}"
            current = []
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(zones, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(zones)} зон(ы) в {args.out}: {list(zones.keys())}")
    try:
        from src import align
        align.save_reference(frame, Path(args.video).name, args.frame)
        print(f"Опорный кадр для совмещения зон: {align.REF_IMAGE} (закоммитьте его вместе с zones.json)")
    except Exception as exc:
        print(f"Опорный кадр не сохранён: {exc}")


if __name__ == "__main__":
    main()