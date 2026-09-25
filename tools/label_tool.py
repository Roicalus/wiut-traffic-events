"""
label_tool.py — ручная разметка событий на одном видео под формат
ground_truth.json хакатона (см. "Output format & interface" в задании).

Запуск:
    python label_tool.py --video samples/C3896.MP4 --out labels/C3896.json

Управление:
    SPACE   — пауза / воспроизведение
    A / D   — шаг на 1 кадр назад/вперёд (на паузе)
    J / L   — прыжок на 1 сек назад/вперёд
    S       — отметить НАЧАЛО события (текущее время)
    E       — отметить КОНЕЦ события -> выбрать класс в консоли,
              событие добавляется в список
    U       — отменить последнее добавленное событие
    Q       — сохранить и выйти

Примечание: перемотка через cap.set(POS_FRAMES) на каждом кадре — не
самая быстрая, но для точной покадровой разметки 4 роликов по 5-6 минут
этого достаточно и она надёжнее, чем ручной подсчёт кадров.
"""
import argparse
import json
import os

import cv2

CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
           "fire_smoke"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(
        "labels", os.path.splitext(os.path.basename(args.video))[0] + ".json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps

    events = []
    pending_start = None
    frame_idx = 0
    playing = False

    cv2.namedWindow("label_tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label_tool", 1280, 720)

    def show():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return
        t = frame_idx / fps
        overlay = frame.copy()
        cv2.putText(overlay, f"t={t:6.2f}s  frame={frame_idx}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if pending_start is not None:
            cv2.putText(overlay, f"START @ {pending_start:.2f}s (E to close)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for i, (s, e, lbl) in enumerate(events[-4:]):
            cv2.putText(overlay, f"[{s:.1f}-{e:.1f}] {lbl}", (10, 90 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow("label_tool", overlay)

    show()
    while True:
        key = cv2.waitKey(0 if not playing else 30) & 0xFF

        if playing:
            frame_idx = min(frame_idx + 1, n_frames - 1)

        if key == ord(' '):
            playing = not playing
        elif key == ord('a') and not playing:
            frame_idx = max(0, frame_idx - 1)
        elif key == ord('d') and not playing:
            frame_idx = min(n_frames - 1, frame_idx + 1)
        elif key == ord('j'):
            frame_idx = max(0, frame_idx - int(fps))
        elif key == ord('l'):
            frame_idx = min(n_frames - 1, frame_idx + int(fps))
        elif key == ord('s'):
            pending_start = frame_idx / fps
            print(f"START отмечен на {pending_start:.2f}s")
        elif key == ord('e'):
            if pending_start is None:
                print("Сначала отметь START клавишей S")
            else:
                t_end = frame_idx / fps
                if t_end <= pending_start:
                    print("END должен быть позже START, пропущено")
                else:
                    print("Классы:", ", ".join(f"{i}:{c}" for i, c in enumerate(CLASSES)))
                    raw = input("Номер класса: ").strip()
                    if raw.isdigit() and 0 <= int(raw) < len(CLASSES):
                        lbl = CLASSES[int(raw)]
                        events.append([round(pending_start, 2), round(t_end, 2), lbl])
                        print(f"Добавлено: [{pending_start:.2f}, {t_end:.2f}, {lbl}]")
                    else:
                        print("Некорректный номер, событие не добавлено")
                pending_start = None
        elif key == ord('u') and events:
            removed = events.pop()
            print(f"Отменено: {removed}")
        elif key == ord('q'):
            break

        show()

    cap.release()
    cv2.destroyAllWindows()

    # предупреждение о пересечениях внутри одного класса — харнесс их
    # всё равно дропнет (оставит более ранний), лучше объединить руками
    by_class = {}
    for s, e, lbl in events:
        for s2, e2 in by_class.get(lbl, []):
            if s < e2 and s2 < e:
                print(f"ВНИМАНИЕ: пересечение в классе {lbl}: [{s2},{e2}] и [{s},{e}]")
        by_class.setdefault(lbl, []).append((s, e))

    events.sort(key=lambda x: x[0])
    data = {
        "video": os.path.basename(args.video),
        "duration": round(duration, 2),
        "fps": fps,
        "events": events,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(events)} событий в {out_path}")


if __name__ == "__main__":
    main()
