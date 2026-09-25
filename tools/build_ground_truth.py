"""
build_ground_truth.py — собирает per-video файлы из labels/*.json (формат
label_tool.py) в один ground_truth.json в формате, который ждёт evaluate.py:

{"video.mp4": {"duration": .., "fps": .., "events": [[s, e, label], ...]}, ...}

Запуск:
    python build_ground_truth.py --labels labels --out ground_truth.json
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--out", default="ground_truth.json")
    args = ap.parse_args()

    result = {}
    for path in sorted(glob.glob(os.path.join(args.labels, "*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        result[d["video"]] = {
            "duration": d["duration"],
            "fps": d["fps"],
            "events": d["events"],
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Собрано {len(result)} видео в {args.out}")


if __name__ == "__main__":
    main()
