"""
build_ground_truth.py — merges the per-video files from labels/*.json (label_tool.py
format) into a single ground_truth.json in the format evaluate.py expects:

{"video.mp4": {"duration": .., "fps": .., "events": [[s, e, label], ...]}, ...}

Usage:
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
    print(f"Merged {len(result)} videos into {args.out}")


if __name__ == "__main__":
    main()
