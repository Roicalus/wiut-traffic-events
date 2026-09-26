"""Zone alignment: a synthetic "background", shifted and slightly rotated."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import align  # noqa: E402


def _scene(w=3840, h=2160, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 70, np.uint8)
    for _ in range(900):          # "buildings/poles/signs" — static texture
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 200)), y + int(rng.integers(20, 200))), c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(10, 80)), c, -1)
    return img


def test_recovers_shift(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "REF_IMAGE", tmp_path / "ref.jpg")
    monkeypatch.setattr(align, "REF_META", tmp_path / "ref.json")
    monkeypatch.setattr(align, "_refs_cache", None)
    ref = _scene()
    align.save_reference(ref, "ref.mp4", 150)

    dx, dy, ang = 80.0, 45.0, 0.4
    R = cv2.getRotationMatrix2D((1920, 1080), ang, 1.0)
    R[:, 2] += (dx, dy)
    moved = cv2.warpAffine(ref, R, (3840, 2160), borderValue=(70, 70, 70))
    # "cars": things that were not in the reference frame
    rng = np.random.default_rng(5)
    for _ in range(60):
        x, y = int(rng.integers(0, 3700)), int(rng.integers(900, 2100))
        cv2.rectangle(moved, (x, y), (x + 140, y + 80), (255, 255, 255), -1)

    M, rep = align.estimate(moved)
    assert rep["status"] == "ok", rep
    zone = {"light_roi": [[2260, 674], [2359, 859]]}
    got = align.transform_zones(zone, M)["light_roi"]
    want = align.transform_zones(zone, R)["light_roi"]
    assert np.abs(got - want).max() < 4.0, (got, want)


def test_no_reference_is_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "REF_IMAGE", tmp_path / "missing.jpg")
    monkeypatch.setattr(align, "_refs_cache", None)
    M, rep = align.estimate(np.zeros((2160, 3840, 3), np.uint8))
    assert M is None and rep["status"] == "no_reference"


def _ref_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(align, "REF_IMAGE", tmp_path / "ref.jpg")
    monkeypatch.setattr(align, "REF_META", tmp_path / "ref.json")
    monkeypatch.setattr(align, "_refs_cache", None)
    ref = _scene()
    align.save_reference(ref, "ref.mp4", 150)
    return ref


def test_recovers_camera_tilt(tmp_path, monkeypatch):
    """The camera was remounted at a different angle: a perspective that shift +
    rotation + scale cannot describe (the stress test missed by 49-262 px)."""
    ref = _ref_scene(tmp_path, monkeypatch)
    src = np.float32([[0, 0], [3840, 0], [3840, 2160], [0, 2160]])
    dst = np.float32([[190, 60], [3650, 60], [3840, 2160], [0, 2160]])
    T = cv2.getPerspectiveTransform(src, dst)
    M, rep = align.estimate(cv2.warpPerspective(ref, T, (3840, 2160), borderValue=(70, 70, 70)))
    assert rep["status"] == "ok" and rep["model"] == "homography", rep
    zone = {"light_roi": [[2260, 674], [2359, 859]]}
    got = align.transform_zones(zone, M)["light_roi"]
    want = align.transform_zones(zone, T)["light_roi"]
    assert np.abs(got - want).max() < 4.0, (got, want)


def test_unrelated_view_is_rejected(tmp_path, monkeypatch):
    """A completely different image — do not align (zones as they are), rather than drawing
    the zones 2000 px off."""
    _ref_scene(tmp_path, monkeypatch)
    M, rep = align.estimate(_scene(seed=7))
    assert M is None and rep["status"] != "ok", rep


def test_lower_resolution_video_aligns(tmp_path, monkeypatch):
    """The same view at 1920x1080: a scale of 0.5 is resolution, not camera motion
    (the plausibility check used to reject it, and the zones stayed at 4K)."""
    ref = _ref_scene(tmp_path, monkeypatch)
    M, rep = align.estimate(cv2.resize(ref, (1920, 1080), interpolation=cv2.INTER_AREA))
    assert rep["status"] == "ok" and abs(rep["scale"] - 1.0) < 0.02, rep
    got = align.transform_zones({"light_roi": [[2260, 674], [2359, 859]]}, M)["light_roi"]
    assert np.abs(got - np.array([[1130, 337], [1179.5, 429.5]])).max() < 3.0, got
