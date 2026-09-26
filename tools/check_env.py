"""check_env.py — checks that the repository is installed correctly.

    python tools/check_env.py

Prints OK / !! for each item and, at the end, what to fix.
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
    bad("Python older than 3.10", "Install Python 3.11 and recreate .venv")
else:
    ok("Python version")

try:
    import numpy
    ok(f"numpy {numpy.__version__}")
except ImportError:
    bad("numpy missing", "pip install -r requirements.txt")

try:
    import cv2
    ok(f"opencv {cv2.__version__}")
    gui = [ln.split(":", 1)[1].strip() for ln in cv2.getBuildInformation().splitlines()
           if ln.strip().startswith("GUI:")]
    if gui and gui[0].upper() != "NONE":
        ok(f"opencv with GUI ({gui[0]}) — label_tool / define_zones will work")
    else:
        bad("opencv without GUI (headless) — label_tool/define_zones windows will not open",
            "pip uninstall -y opencv-python-headless opencv-python && pip install opencv-python")
except ImportError:
    bad("opencv missing", "pip install -r requirements.txt")

try:
    from importlib.metadata import distributions
    cv_pkgs = sorted({d.metadata["Name"].lower() for d in distributions()
                      if (d.metadata["Name"] or "").lower().startswith("opencv")})
    if len(cv_pkgs) > 1:
        bad(f"several OpenCV packages installed at once: {cv_pkgs}",
            'pip uninstall -y ' + " ".join(cv_pkgs) + ' && pip install "opencv-python>=4.8,<5"')
    import cv2 as _cv2
    if int(_cv2.__version__.split(".")[0]) >= 5:
        bad(f"OpenCV {_cv2.__version__}: the code is tested on 4.x",
            'pip uninstall -y opencv-python opencv-python-headless && pip install "opencv-python>=4.8,<5"')
except Exception:
    pass

try:
    import shutil
    import torch
    cuda = torch.cuda.is_available()
    ok(f"torch {torch.__version__}, CUDA: {'yes, ' + torch.cuda.get_device_name(0) if cuda else 'no (will use CPU)'}")
    if "+cu" in torch.__version__ and not cuda and shutil.which("nvidia-smi") is None:
        bad("the CUDA build of torch is installed but there is no NVIDIA GPU — an extra 2.4 GB and a slow import",
            "pip uninstall -y torch torchvision && pip install torch torchvision")
except ImportError:
    bad("torch missing", "pip install -r requirements.txt")

try:
    import ultralytics
    ok(f"ultralytics {ultralytics.__version__}")
except ImportError:
    bad("ultralytics missing", "pip install -r requirements.txt")

try:
    import lap  # noqa: F401
    ok("lap (required by ByteTrack)")
except ImportError:
    bad("lap missing", "pip install lap")

for w in ("yolo11s.pt", "yolo11n.pt"):
    p = ROOT / "weights" / w
    if p.exists() and p.stat().st_size > 1_000_000:
        ok(f"weights/{w} ({p.stat().st_size / 1e6:.1f} MB)")
    else:
        bad(f"weights/{w} missing", "Download the weights: bash weights/download.sh or the setup script")

try:
    zones = json.loads((ROOT / "zones.json").read_text())
    ok(f"zones.json: {len(zones)} zones")
except Exception as exc:
    bad(f"zones.json is unreadable ({exc})", "Put zones.json back in the repository root")

samples = ROOT / "samples"
vids = sorted(p.name for p in samples.glob("*")) if samples.exists() else []
vids = [v for v in vids if v.lower().endswith(".mp4")]
if vids:
    ok(f"samples/: {len(vids)} videos ({', '.join(vids[:4])}{'…' if len(vids) > 4 else ''})")
else:
    bad("samples/ is empty or missing", "Put the sample videos (.mp4) into the samples/ folder")

try:
    sys.path.insert(0, str(ROOT))
    import solution
    ok(f"solution.py imports, CLASSES = {solution.CLASSES}")
except Exception as exc:
    bad(f"solution.py does not import: {exc!r}", "See the error text above")

print()
if problems:
    print("What to fix:")
    for i, fix in enumerate(dict.fromkeys(problems), 1):
        print(f"  {i}. {fix}")
    sys.exit(1)
print("All set.")
