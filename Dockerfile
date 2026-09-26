# Alternative to requirements.txt (allowed by the task):
#   docker build -t team .
#   docker run --rm --gpus all -v /data/test:/data/test -v "$PWD/out":/out team \
#       python run_submission.py --videos /data/test --out /out/predictions.json
FROM python:3.11-slim

# opencv-python (required by ultralytics) needs libGL and glib even without a display
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# weights are already in the repository; if missing, download them at build time (internet is available only here)
RUN bash weights/download.sh

ENV YOLO_OFFLINE=1 \
    YOLO_VERBOSE=False \
    PYTHONUNBUFFERED=1
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/out/predictions.json"]
