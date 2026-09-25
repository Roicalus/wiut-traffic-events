"""check_env.py — проверка, что репозиторий установлен правильно.

    python tools/check_env.py

Печатает OK / !! по каждому пункту и в конце — что исправить.
"""
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
problems = []


def ok(msg):
    print(f"  OK  {msg}")


def bad(msg, fix):
    print(f"  !!  {msg}")
    problems.append(fix)


print(f"Python {platform.python_version()} ({sys.executable})")
if sys.version_info < (3, 10):
    bad("Python старше 3.10", "Установите Python 3.11 и пересоздайте .venv")
else:
    ok("версия Python")

try:
    import numpy
    ok(f"numpy {numpy.__version__}")
except ImportError:
    bad("нет numpy", "pip install -r requirements.txt")

try:
    import cv2
    ok(f"opencv {cv2.__version__}")
    gui = [ln.split(":", 1)[1].strip() for ln in cv2.getBuildInformation().splitlines()
           if ln.strip().startswith("GUI:")]
    if gui and gui[0].upper() != "NONE":
        ok(f"opencv с GUI ({gui[0]}) — label_tool / define_zones будут работать")
    else:
        bad("opencv без GUI (headless) — окна label_tool/define_zones не откроются",
            "pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python")
except ImportError:
    bad("нет opencv", "pip install -r requirements.txt")

try:
    from importlib.metadata import distributions
    cv_pkgs = sorted({d.metadata["Name"].lower() for d in distributions()
                      if (d.metadata["Name"] or "").lower().startswith("opencv")})
    if len(cv_pkgs) > 1:
        bad(f"установлено несколько пакетов OpenCV одновременно: {cv_pkgs}",
            'pip uninstall -y ' + " ".join(cv_pkgs) + ' && pip install "opencv-python>=4.8,<5"')
    import cv2 as _cv2
    if int(_cv2.__version__.split(".")[0]) >= 5:
        bad(f"OpenCV {_cv2.__version__}: код проверен на 4.x",
            'pip uninstall -y opencv-python opencv-python-headless && pip install "opencv-python>=4.8,<5"')
except Exception:
    pass

try:
    import shutil
    import torch
    cuda = torch.cuda.is_available()
    ok(f"torch {torch.__version__}, CUDA: {'да, ' + torch.cuda.get_device_name(0) if cuda else 'нет (будет CPU)'}")
    if "+cu" in torch.__version__ and not cuda and shutil.which("nvidia-smi") is None:
        bad("стоит CUDA-версия torch, но видеокарты NVIDIA нет — лишние 2.4 ГБ и медленный импорт",
            "pip uninstall -y torch torchvision && pip install torch torchvision")
except ImportError:
    bad("нет torch", "pip install -r requirements.txt")

try:
    import ultralytics
    ok(f"ultralytics {ultralytics.__version__}")
except ImportError:
    bad("нет ultralytics", "pip install -r requirements.txt")

try:
    import lap  # noqa: F401
    ok("lap (нужен ByteTrack)")
except ImportError:
    bad("нет lap", "pip install lap")

for w in ("yolo11s.pt", "yolo11n.pt"):
    p = ROOT / "weights" / w
    if p.exists() and p.stat().st_size > 1_000_000:
        ok(f"weights/{w} ({p.stat().st_size / 1e6:.1f} МБ)")
    else:
        bad(f"нет weights/{w}", "Скачайте веса: bash weights/download.sh или setup-скрипт")

try:
    zones = json.loads((ROOT / "zones.json").read_text())
    ok(f"zones.json: {len(zones)} зон")
except Exception as exc:
    bad(f"zones.json не читается ({exc})", "Верните zones.json в корень репозитория")

samples = ROOT / "samples"
vids = sorted(p.name for p in samples.glob("*")) if samples.exists() else []
vids = [v for v in vids if v.lower().endswith(".mp4")]
if vids:
    ok(f"samples/: {len(vids)} видео ({', '.join(vids[:4])}{'…' if len(vids) > 4 else ''})")
else:
    bad("samples/ пуст или отсутствует", "Положите сэмпл-видео (.mp4) в папку samples/")

try:
    sys.path.insert(0, str(ROOT))
    import solution
    ok(f"solution.py импортируется, CLASSES = {solution.CLASSES}")
except Exception as exc:
    bad(f"solution.py не импортируется: {exc!r}", "Смотрите текст ошибки выше")

print()
if problems:
    print("Что исправить:")
    for i, fix in enumerate(dict.fromkeys(problems), 1):
        print(f"  {i}. {fix}")
    sys.exit(1)
print("Всё готово.")
