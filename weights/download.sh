#!/usr/bin/env bash
# Веса детекторов (Ultralytics YOLO11, COCO, AGPL-3.0). Они уже лежат в
# репозитории; скрипт нужен, только если их нет: запускается один раз, с
# интернетом, до офлайн-прогона. Хэши — тех самых файлов, на которых
# получен predictions_samples.json.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/ultralytics/assets/releases/download/v8.3.0
declare -A SHA=(
  [yolo11s.pt]=85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5
  [yolo11n.pt]=0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
)
for f in yolo11s.pt yolo11n.pt; do
  if [ ! -s "$f" ]; then
    echo ">> скачиваю $f"
    curl -fL --retry 3 -o "$f" "$BASE/$f"
  fi
  echo "${SHA[$f]}  $f" | sha256sum -c -
done
