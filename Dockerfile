# Альтернатива requirements.txt (разрешена заданием):
#   docker build -t team .
#   docker run --rm --gpus all -v /data/test:/data/test -v "$PWD/out":/out team \
#       python run_submission.py --videos /data/test --out /out/predictions.json
FROM python:3.11-slim

# opencv-python (его требует ultralytics) нужны libGL и glib даже без экрана
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# веса уже в репозитории; если их нет — скачиваем при сборке (интернет есть только тут)
RUN bash weights/download.sh

ENV YOLO_OFFLINE=1 \
    YOLO_VERBOSE=False \
    PYTHONUNBUFFERED=1
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/out/predictions.json"]
