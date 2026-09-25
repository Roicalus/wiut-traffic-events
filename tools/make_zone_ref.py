"""make_zone_ref.py — опорные кадры для совмещения зон (src/align.py).

Основной опорный кадр должен быть ТЕМ ЖЕ кадром, на котором рисовался
zones.json (define_zones.py по умолчанию берёт --frame 150):

    python tools/make_zone_ref.py --video samples/C3896.MP4 --frame 150

Дополнительные опорные кадры — то же место при другом освещении (закат,
сумерки): тёмное видео сопоставляется с тёмным опорным кадром. Переход
"основной -> этот" считается автоматически:

    python tools/make_zone_ref.py --video samples/C3905.MP4 --frame 300 --extra dusk

Результат: zones_ref.jpg, zones_ref_<имя>.jpg, zones_ref.json — закоммитить.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import align  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=150)
    ap.add_argument("--extra", default=None, help="имя дополнительного опорного кадра (dusk, sunset, ...)")
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"не удалось прочитать кадр {args.frame} из {args.video}")
    if args.extra:
        rep = align.add_extra_reference(frame, args.extra, Path(args.video).name, args.frame)
        print(f"Добавлен опорный кадр {args.extra}: {rep}")
    else:
        align.save_reference(frame, Path(args.video).name, args.frame)
        print(f"Сохранено: {align.REF_IMAGE} и {align.REF_META}")
    print("Проверьте совмещение: python tools/check_alignment.py --videos samples")


if __name__ == "__main__":
    main()
