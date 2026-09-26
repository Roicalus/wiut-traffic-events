"""align_stress.py — how robust zone alignment is to camera displacement.

Real sample frames + a synthetic "different camera position": shift,
rotation, scale, tilt (perspective), occlusion of part of the frame, blur,
noise, darkening. The alignment is exactly what the submission uses (src/align.estimate,
with plausibility checks). Error — the maximum miss at the corners of the zones
light_roi, stop_line, crossing_far, crossing_near, px in 4K.

Honest mode (default): for a frame of video X, the reference frame taken on X is
removed from the reference-frame bank — as for the hidden test, which is not in
the bank.

    python tools/align_stress.py --videos samples
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import align  # noqa: E402
from src.pipeline import get_zones  # noqa: E402

W0, H0 = 3840, 2160
PROBE_ZONES = ("light_roi", "stop_line", "crossing_far", "crossing_near")


def perturbations():
    """(name, T 3x3 — the true extra camera displacement, frame corruption | None)."""
    def aff(rot=0.0, scale=1.0, dx=0.0, dy=0.0):
        R = cv2.getRotationMatrix2D((W0 / 2, H0 / 2), rot, scale)
        R[:, 2] += (dx, dy)
        return np.vstack([R, [0, 0, 1]])

    def tilt(k):
        src = np.float32([[0, 0], [W0, 0], [W0, H0], [0, H0]])
        dst = np.float32([[W0 * k, H0 * k / 2], [W0 * (1 - k), H0 * k / 2], [W0, H0], [0, H0]])
        return cv2.getPerspectiveTransform(src, dst)

    def occlude(frac):
        def f(img):
            img = img.copy()
            img[:, : int(img.shape[1] * frac)] = 40
            return img
        return f

    def gamma(g):
        return lambda img: np.clip(255.0 * (img / 255.0) ** g, 0, 255).astype(np.uint8)

    I = np.eye(3)
    return [
        ("as is", I, None),
        ("shift 300x150", aff(dx=300, dy=150), None),
        ("shift 500x250", aff(dx=500, dy=250), None),
        ("rotate 4°", aff(rot=4), None),
        ("rotate 8°", aff(rot=8), None),
        ("scale 0.8", aff(scale=0.8), None),
        ("scale 1.25", aff(scale=1.25), None),
        ("tilt 3%", tilt(0.03), None),
        ("tilt 8%", tilt(0.08), None),
        ("shift+rotate+tilt", aff(rot=3, dx=250, dy=-120) @ tilt(0.04), None),
        ("occluded 30%", I, occlude(0.3)),
        ("blur", I, lambda img: cv2.GaussianBlur(img, (0, 0), 4)),
        ("noise", I, lambda img: np.clip(img + np.random.default_rng(0).normal(0, 20, img.shape), 0, 255)
         .astype(np.uint8)),
        ("darker x2.5", I, gamma(2.5)),
        ("brighter", I, gamma(0.5)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "samples"))
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--with-own-reference", action="store_true",
                    help="do not remove the reference frame of the same video (dishonest, for comparison)")
    args = ap.parse_args()
    zones = get_zones()
    probe = np.vstack([zones[n] for n in PROBE_ZONES]).astype(np.float64)
    refs_all = align.load_reference()
    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() == ".mp4")

    table = collections.defaultdict(list)
    for v in videos:
        frame = align.first_frame(v, args.frame)
        refs = refs_all if args.with_own_reference else [r for r in refs_all if r["video"] != v.name
                                                         or r["name"] == "main"]
        if v.name == refs_all[0]["video"] and not args.with_own_reference:
            # video of the main reference frame: a different frame of the same clip — the most honest possible
            frame = align.first_frame(v, args.frame + 3000)
        H_v, rep = align.estimate(frame, refs=refs)
        if H_v is None:
            print(f"[{v.name}] base alignment failed: {rep}")
            continue
        truth_v = cv2.perspectiveTransform(probe[None], H_v)[0]
        for name, T, spoil in perturbations():
            img = frame if np.allclose(T, np.eye(3)) else cv2.warpPerspective(frame, T, (W0, H0))
            if spoil is not None:
                img = spoil(img)
            H, rep = align.estimate(img, refs=refs)
            truth = cv2.perspectiveTransform(truth_v[None], T)[0]
            if H is None:
                table[name].append((v.stem, None, rep.get("status")))
            else:
                err = float(np.abs(cv2.perspectiveTransform(probe[None], H)[0] - truth).max())
                table[name].append((v.stem, err, f"{rep['model']}/{rep['ref']}"))

    print(f"{'displacement':24s} " + "  ".join(f"{v.stem:>18s}" for v in videos))
    for name, rows in table.items():
        cells = [f"{'—':>6s} {st:>11s}" if e is None else f"{e:6.1f} {st:>11s}" for _, e, st in rows]
        print(f"{name:24s} " + "  ".join(cells))
    errs = [e for rows in table.values() for _, e, _ in rows if e is not None]
    n = sum(len(r) for r in table.values())
    print(f"\naligned {len(errs)}/{n}, error px: median {np.median(errs):.1f}, "
          f"90% {np.percentile(errs, 90):.1f}, max {max(errs):.1f}")
    print("Error is relative to the base alignment of the same frame (which is itself accurate: "
          "see tools/check_alignment.py).")


if __name__ == "__main__":
    main()
