"""build_space.py — собирает папку для Hugging Face Space с живым демо.

    python tools/build_space.py --out ../space
    cd ../space && git push                   # в репозиторий Space (см. README)

В Space попадает ровно код сабмита (solution.py, src/, веса, зоны и опорные
кадры) плюс demo/app.py как app.py. requirements.txt — из demo/ (torch в
CPU-сборке, Gradio); packages.txt — системные библиотеки для OpenCV.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["solution.py", "zones.json", "zones_ref.json"] + [p.name for p in ROOT.glob("zones_ref*.jpg")]
DIRS = ["src", "weights"]
SPACE_README = """---
title: Traffic events — live demo
emoji: 🚦
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: false
---

Live demo of our WIUT Hackathon 2026 CV-track submission: upload an .mp4 from the
intersection camera and get the detected traffic events, the accident-risk curve and an
annotated video back. Code: {repo}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="папка Space (создаётся/обновляется)")
    ap.add_argument("--repo", default="(GitHub link)", help="ссылка на репозиторий для README Space")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in DIRS:
        shutil.rmtree(out / name, ignore_errors=True)
        shutil.copytree(ROOT / name, out / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in FILES:
        shutil.copy2(ROOT / name, out / name)
    app = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
    # в Space app.py лежит в корне, рядом с solution.py
    app = app.replace("ROOT = Path(__file__).resolve().parent.parent", "ROOT = Path(__file__).resolve().parent")
    app = app.replace('server_name=os.environ.get("DEMO_HOST", "127.0.0.1")', 'server_name=os.environ.get("DEMO_HOST", "0.0.0.0")')
    (out / "app.py").write_text(app, encoding="utf-8", newline="\n")
    shutil.copy2(ROOT / "demo" / "requirements.txt", out / "requirements.txt")
    (out / "packages.txt").write_text("libgl1\nlibglib2.0-0\n", encoding="utf-8", newline="\n")
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n",
                                        encoding="utf-8", newline="\n")
    import gradio
    (out / "README.md").write_text(SPACE_README.format(gradio=gradio.__version__, repo=args.repo),
                                   encoding="utf-8", newline="\n")
    print(f"Space собран: {out.resolve()}")


if __name__ == "__main__":
    main()
