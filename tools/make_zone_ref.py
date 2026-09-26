"""make_zone_ref.py — reference frames for zone alignment (src/align.py).

The main reference frame must be the SAME frame that zones.json was drawn on
(define_zones.py uses --frame 150 by default):

    python tools/make_zone_ref.py --video samples/C3896.MP4 --frame 150

Extra reference frames show the same place under different lighting (sunset,
dusk): a dark video is matched against a dark reference frame. The
"main -> this one" transform is computed automatically:

    python tools/make_zone_ref.py --video samples/C3905.MP4 --frame 300 --extra dusk

Output: zones_ref.jpg, zones_ref_<name>.jpg, zones_ref.json — commit them.
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
    ap.add_argument("--extra", default=None, help="name of the extra reference frame (dusk, sunset, ...)")
    args = ap.parse_args()
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame} from {args.video}")
    if args.extra:
        rep = align.add_extra_reference(frame, args.extra, Path(args.video).name, args.frame)
        print(f"Added reference frame {args.extra}: {rep}")
    else:
        align.save_reference(frame, Path(args.video).name, args.frame)
        print(f"Saved: {align.REF_IMAGE} and {align.REF_META}")
    print("Check the alignment: python tools/check_alignment.py --videos samples")


if __name__ == "__main__":
    main()
