"""align.py — совмещение зон с конкретным видео.

Камера "та же", но между записями она смещается: относительно C3896 (по
нему рисовались зоны) C3902 сдвинут на 140x77 px, повёрнут на 1° и
приближен на 1.5%, C3905 — на 41x42 px, 1.1°, 1.5%. Скрытый тест снят той
же камерой, но другими записями, поэтому зоны переводятся в координаты
КАЖДОГО видео по его кадрам.

Метод:
  * CLAHE -> SIFT на уменьшенных кадрах (WORK_WIDTH) -> тест Лоу;
  * модель — гомография (RANSAC): она описывает и наклон камеры, если её
    перевесили под другим углом. Сдвиг + поворот + масштаб на
    стресс-тесте с наклоном 2-8% промахивался на 49-262 px, гомография —
    на 2-4 px; даже на самих сэмплах гомография точнее (1-4 px против
    11-15). Если соответствий мало или гомография вырождена — откат на
    сдвиг + поворот + масштаб, если и он неправдоподобен — зоны как есть;
  * банк опорных кадров: основной (день, zones_ref.jpg) и дополнительные
    (закат, сумерки) с заранее посчитанным переходом "основной -> этот
    опорный". Тёмное видео сопоставляется с тёмным опорным кадром;
  * по нескольким кадрам видео: сетка контрольных точек проецируется
    каждой оценкой, по точкам берётся медиана, по медиане строится итоговая
    гомография. Камера качается в пределах ролика (~10 px), а одиночный
    кадр может поймать автобус перед фоном. Если оценки расходятся сильнее
    MOVED_WARN_PX, в отчёт пишется, что камеру сдвигали по ходу ролика.

Стресс-тест (tools/align_stress.py) на реальных кадрах 4 сэмплов: сдвиг до
500x250 px, поворот до 8°, масштаб 0.8-1.25, наклон до 8%, перекрытие 30%
кадра, размытие, шум, затемнение — ошибка центра light_roi <= 7 px.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF_IMAGE = ROOT / "zones_ref.jpg"
REF_META = ROOT / "zones_ref.json"
WORK_WIDTH = 1280          # совмещаем на уменьшенных кадрах
MIN_INLIERS = 40           # подобие
MIN_INLIERS_H = 60         # гомография: 8 степеней свободы — нужно больше опоры
MIN_INLIER_RATIO_H = 0.2
RATIO = 0.8                # тест Лоу для SIFT
N_FEATURES = 4000          # перебор соответствий квадратичен: 8000 — 3.4 с на кадр с банком из 3
RANSAC_PX = 3.0            # порог RANSAC в px уменьшенного кадра
SAMPLE_FRACS = (0.0, 0.5, 0.9)   # кадры видео для оценки (доли длительности)
SHORT_VIDEO_SEC = 60.0           # короче — один кадр: перемотка 4K стоит ~2.5 с на кадр
# Правдоподобие "та же камера": на стресс-тесте оценки точны (<= 7 px) далеко
# за прежними границами 8% / 3° / 5%; неправдоподобное — это ошибка сопоставления
# (глубокая ночь: 22 инлаера, промах на 2000+ px), а не настоящее смещение.
MAX_SHIFT_FRAC = 0.25
MAX_ROT_DEG = 10.0
MAX_SCALE_DEV = 0.35             # приближение камеры 1.25 + своё 1.3% ещё проходит
MAX_PERSPECTIVE_AREA_DEV = 0.5   # площадь кадра / масштаб^2: 0.5-1.5 (вырожденная перспектива — нет)
MOVED_WARN_PX = 40.0
CONFIDENT_INLIERS = 400          # столько инлаеров — остальные опорные кадры не перебираем

_refs_cache = None


# ---------------------------------------------------------------- опорные кадры
def save_reference(frame_bgr, video_name: str, frame_idx: int) -> None:
    """Сохраняет ОСНОВНОЙ опорный кадр (уменьшенный) и метаданные рядом с
    zones.json. Уже добавленные дополнительные опорные кадры сохраняются
    (см. add_extra_reference)."""
    global _refs_cache
    h, w = frame_bgr.shape[:2]
    cv2.imwrite(str(REF_IMAGE), _small(frame_bgr), [cv2.IMWRITE_JPEG_QUALITY, 92])
    old = json.loads(REF_META.read_text()) if REF_META.exists() else {}
    REF_META.write_text(json.dumps({"video": video_name, "frame": frame_idx,
                                    "full_width": w, "full_height": h,
                                    "extra": old.get("extra", [])}, indent=1))
    _refs_cache = None


def add_extra_reference(frame_bgr, name: str, video_name: str, frame_idx: int) -> dict:
    """Дополнительный опорный кадр (другое освещение). Переход "основной ->
    этот" считается здесь же, по основному опорному кадру; если совмещение
    не удалось — кадр не добавляется."""
    global _refs_cache
    H, rep = estimate(frame_bgr, refs=_load_refs()[:1])
    if H is None:
        raise RuntimeError(f"дополнительный опорный кадр не совместился с основным: {rep}")
    h, w = frame_bgr.shape[:2]
    image = f"zones_ref_{name}.jpg"
    cv2.imwrite(str(REF_META.parent / image), _small(frame_bgr), [cv2.IMWRITE_JPEG_QUALITY, 92])
    meta = json.loads(REF_META.read_text())
    extra = [e for e in meta.get("extra", []) if e["name"] != name]
    extra.append({"name": name, "image": image, "video": video_name, "frame": frame_idx,
                  "full_width": w, "full_height": h, "H_from_main": np.asarray(H).tolist()})
    meta["extra"] = extra
    REF_META.write_text(json.dumps(meta, indent=1))
    _refs_cache = None
    return rep


def _small(frame_bgr):
    h, w = frame_bgr.shape[:2]
    return cv2.resize(frame_bgr, (WORK_WIDTH, int(round(h * WORK_WIDTH / w))), interpolation=cv2.INTER_AREA)


def _features(gray):
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.SIFT_create(nfeatures=N_FEATURES).detectAndCompute(gray, None)


def load_reference():
    """Все опорные кадры и их SIFT-признаки (кэшируются на процесс)."""
    return _load_refs()


def _load_refs() -> list[dict]:
    """[{name, gray, kp, des, full_width, H_from_main}], основной — первым."""
    global _refs_cache
    if _refs_cache is None:
        _refs_cache = []
        if REF_IMAGE.exists():
            meta = json.loads(REF_META.read_text()) if REF_META.exists() else {}
            entries = [{"name": "main", "path": REF_IMAGE, "H_from_main": np.eye(3).tolist(),
                        "full_width": meta.get("full_width"), "video": meta.get("video")}]
            entries += [dict(e, path=REF_META.parent / e["image"]) for e in meta.get("extra", [])]
            for e in entries:
                path = e["path"]
                if not path.exists():
                    continue
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                kp, des = _features(gray)
                _refs_cache.append({"name": e["name"], "gray": gray, "kp": kp, "des": des,
                                    "brightness": float(gray.mean()),
                                    "full_width": float(e.get("full_width") or gray.shape[1]),
                                    "video": e.get("video"),
                                    "H_from_main": np.asarray(e["H_from_main"], np.float64)})
    return _refs_cache


# ---------------------------------------------------------------- оценка
def _to3(M):
    return np.vstack([M, [0.0, 0.0, 1.0]])


def _describe(H, w, h):
    """Сдвиг центра кадра, поворот и масштаб гомографии вокруг центра — для
    отчёта и проверки правдоподобия."""
    c = np.array([[[w / 2, h / 2]]])
    d = 50.0
    p = cv2.perspectiveTransform(np.array([[[w / 2, h / 2], [w / 2 + d, h / 2]]]), H)[0]
    cc = cv2.perspectiveTransform(c, H)[0, 0]
    vx = (p[1] - p[0]) / d
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float64)
    area = cv2.contourArea(cv2.perspectiveTransform(corners, H)[0].astype(np.float32)) / (w * h)
    return {"dx": round(float(cc[0] - w / 2), 1), "dy": round(float(cc[1] - h / 2), 1),
            "rot_deg": round(float(np.degrees(np.arctan2(vx[1], vx[0]))), 2),
            "scale": round(float(np.hypot(*vx)), 4), "area": round(float(area), 3)}


def _plausible(desc, w):
    return (abs(desc["scale"] - 1) <= MAX_SCALE_DEV and abs(desc["rot_deg"]) <= MAX_ROT_DEG
            and max(abs(desc["dx"]), abs(desc["dy"])) <= MAX_SHIFT_FRAC * w
            and abs(desc["area"] / desc["scale"] ** 2 - 1) <= MAX_PERSPECTIVE_AREA_DEV)


def _describe_camera(H, ref_w, ref_h, w):
    """_describe смещения КАМЕРЫ: H переводит опорный кадр (ref_w) в кадр видео
    (w), и у видео другого разрешения масштаб w/ref_w — это не движение
    камеры. Сравниваем в масштабе опорного кадра."""
    k = w / ref_w
    return _describe(np.diag([1 / k, 1 / k, 1.0]) @ H, ref_w, ref_h)


def _estimate_one(ref, gray, kp_f, des_f, w, h):
    """Гомография "опорный кадр ref -> этот кадр" в полных px, или None."""
    if ref["des"] is None or des_f is None or len(kp_f) < MIN_INLIERS:
        return None, {"status": "few_features"}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ref["des"], des_f, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None, {"status": "few_matches", "matches": len(good)}
    k_ref = ref["full_width"] / ref["gray"].shape[1]    # уменьшенный -> полный опорный
    k_frm = w / gray.shape[1]                            # уменьшенный -> полный кадр видео
    src = np.float32([ref["kp"][m.queryIdx].pt for m in good]) * k_ref
    dst = np.float32([kp_f[m.trainIdx].pt for m in good]) * k_frm
    thr = RANSAC_PX * k_frm
    cv2.setRNGSeed(0)   # RANSAC случайный: фиксируем, чтобы два прогона совпадали
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, thr, maxIters=5000, confidence=0.999)
    n_h = int(inl.sum()) if inl is not None else 0
    ref_w, ref_h = ref["full_width"], ref["gray"].shape[0] * k_ref
    if H is not None and n_h >= MIN_INLIERS_H and n_h / len(good) >= MIN_INLIER_RATIO_H:
        desc = _describe_camera(H, ref_w, ref_h, w)
        if _plausible(desc, ref_w):
            return H, dict(desc, status="ok", model="homography", inliers=n_h, matches=len(good))
    cv2.setRNGSeed(0)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr,
                                         maxIters=5000, confidence=0.999)
    n_s = int(inl.sum()) if inl is not None else 0
    if M is None or n_s < MIN_INLIERS:
        return None, {"status": "ransac_failed", "inliers": max(n_h, n_s), "matches": len(good)}
    H = _to3(M)
    desc = _describe_camera(H, ref_w, ref_h, w)
    if not _plausible(desc, ref_w):
        return None, dict(desc, status="implausible", inliers=n_s, matches=len(good))
    return H, dict(desc, status="ok", model="similarity", inliers=n_s, matches=len(good))


def estimate(frame_bgr, refs=None) -> tuple[np.ndarray | None, dict]:
    """Гомография 3x3 (координаты зон, т.е. основного опорного кадра в полном
    разрешении -> координаты этого кадра) и отчёт. Перебираются все опорные
    кадры, берётся оценка с наибольшим числом инлаеров. None — не удалось."""
    refs = _load_refs() if refs is None else refs
    if not refs:
        return None, {"status": "no_reference", "hint": "нет zones_ref.jpg — tools/make_zone_ref.py"}
    h, w = frame_bgr.shape[:2]
    small = _small(frame_bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = _features(gray)
    best, best_rep, fails = None, None, []
    # сначала — ближайший по освещению опорный кадр: обычно его и хватает
    brightness = float(gray.mean())
    for ref in sorted(refs, key=lambda r: abs(r["brightness"] - brightness)):
        H, rep = _estimate_one(ref, gray, kp_f, des_f, w, h)
        if H is None:
            fails.append(dict(rep, ref=ref["name"]))
            continue
        if best is None or rep["inliers"] > best_rep["inliers"]:
            best, best_rep = H @ ref["H_from_main"], dict(rep, ref=ref["name"], ref_video=ref["video"])
        if best_rep is not None and best_rep["inliers"] >= CONFIDENT_INLIERS:
            break   # похожий по свету опорный кадр даёт сотни инлаеров — дальше не ищем
    if best is None:
        return None, fails[0] if len(fails) == 1 else {"status": fails[0]["status"], "tried": fails}
    best /= best[2, 2]
    main = refs[0] if refs[0]["name"] == "main" else _load_refs()[0]
    best_rep.update(_describe_camera(best, main["full_width"],
                                     main["gray"].shape[0] * main["full_width"] / main["gray"].shape[1], w))
    return best, best_rep


# ---------------------------------------------------------------- видео
def first_frame(video_path, frame_idx: int = 10):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def transform_zones(zones: dict, M: np.ndarray | None) -> dict:
    """zones: имя -> массив/список точек; M — 2x3 (аффинная) или 3x3
    (гомография). Возвращает новый dict с np.float32."""
    out = {}
    for name, pts in zones.items():
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if M is not None:
            M = np.asarray(M, np.float64)
            arr = cv2.perspectiveTransform(arr[None], M if M.shape == (3, 3) else _to3(M))[0]
        out[name] = arr.astype(np.float32)
    return out


def _control_grid(w, h, n=6):
    xs, ys = np.linspace(0.05 * w, 0.95 * w, n), np.linspace(0.05 * h, 0.95 * h, n)
    return np.array([[x, y] for y in ys for x in xs], np.float64)


def estimate_video(video_path) -> tuple[np.ndarray | None, dict]:
    """Оценки по кадрам SAMPLE_FRACS, сведённые через медиану проекций сетки
    контрольных точек (устойчиво к одной плохой оценке и не требует, чтобы
    все кадры дали одну и ту же модель)."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    Hs, reports, size = [], [], None
    for frac in (SAMPLE_FRACS if n / fps >= SHORT_VIDEO_SEC else SAMPLE_FRACS[:1]):
        frame = first_frame(video_path, max(10, min(int(frac * n), n - 10)))
        if frame is None:
            continue
        size = frame.shape[1], frame.shape[0]
        H, rep = estimate(frame)
        reports.append(rep)
        if H is not None:
            Hs.append(H)
    if not reports:
        return None, {"status": "no_frame"}
    if not Hs:
        return None, reports[0]
    refs = _load_refs()
    ref_w = refs[0]["full_width"]
    ref_h = refs[0]["gray"].shape[0] * ref_w / refs[0]["gray"].shape[1]
    grid = _control_grid(ref_w, ref_h)
    proj = np.stack([cv2.perspectiveTransform(grid[None], H)[0] for H in Hs])
    median = np.median(proj, axis=0)
    H, _ = cv2.findHomography(grid, median, 0)
    report = dict(max((r for r in reports if r["status"] == "ok"), key=lambda r: r["inliers"]))
    report.update(_describe_camera(H, ref_w, ref_h, size[0]))
    report["frames_ok"] = f"{len(Hs)}/{len(reports)}"
    spread = float(np.abs(proj - median).max()) if len(Hs) > 1 else 0.0
    report["spread_px"] = round(spread, 1)
    if spread > MOVED_WARN_PX:
        report["warning"] = "оценки по кадрам расходятся — камеру, похоже, сдвигали по ходу ролика"
    return H, report


def aligned_zones(video_path, zones: dict) -> tuple[dict, dict]:
    M, report = estimate_video(video_path)
    if M is None:
        # совмещение не удалось — но если разрешение другое, зоны хотя бы масштабируем
        refs = _load_refs()
        cap = cv2.VideoCapture(str(video_path))
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap.release()
        ref_w = refs[0]["full_width"] if refs else None
        if ref_w and w and abs(w / ref_w - 1.0) > 1e-3:
            k = w / ref_w
            M = np.array([[k, 0.0, 0.0], [0.0, k, 0.0]], np.float64)
            report = dict(report, rescaled=round(k, 4))
    return transform_zones(zones, M), report
