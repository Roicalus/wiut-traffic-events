#!/usr/bin/env bash
# Development setup on a laptop (Linux / macOS).  Run:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 is required"'

[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1 && [ "$(uname)" = "Linux" ]; then
  echo ">> NVIDIA found — installing torch with CUDA"
  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
fi
pip install -r requirements.txt
pip install pytest

# old environments may still have opencv-python-headless — it shadows the GUI build
pip uninstall -y opencv-python-headless opencv-python >/dev/null 2>&1 || true
pip install "opencv-python>=4.8,<5"

bash weights/download.sh
mkdir -p samples labels cache debug

python -m pytest tests -q
python tools/check_env.py || true
echo
echo "Done. In a new terminal, first run:  source .venv/bin/activate"
