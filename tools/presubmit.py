"""presubmit.py — проверка репозитория по требованиям сабмишена.

    python tools/presubmit.py                         # статические проверки
    python tools/presubmit.py --determinism samples/C3905.MP4   # + два прогона подряд

Каждый пункт: OK / WARN / FAIL. Код возврата 1, если есть FAIL.
Запускайте перед финальным коммитом и перед тегом.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# sha256 файлов стартового набора организаторов: они должны лежать без изменений
STARTER_KIT = {
    "run_submission.py": "a47b494afae14432a65166b43cd5f2a278408ce661aecf0fb86b6fa05a06c204",
    "evaluate.py": "111c6fa04709c9f1df4ea3db4bede749953b2c27bdc5679c2ef56794637573b5",
}
WEIGHTS = {
    "yolo11s.pt": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
}
MAX_WEIGHTS_BYTES = 5 * 1024 ** 3
TIME_MARGIN = 0.7   # хотим total_sec <= 70% бюджета — запас на более медленную машину

results: list[tuple[str, str, str]] = []


def report(level, item, msg=""):
    results.append((level, item, msg))
    print(f"  {level:<4} {item}" + (f" — {msg}" if msg else ""))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def check_layout():
    print("\n[1] Структура репозитория")
    for f in ("solution.py", "run_submission.py", "evaluate.py", "requirements.txt", "README.md",
              "predictions_samples.json", "zones.json"):
        report("OK" if (ROOT / f).exists() else "FAIL", f, "" if (ROOT / f).exists() else "нет файла")
    report("OK" if (ROOT / "src").is_dir() else "FAIL", "src/")
    report("OK" if (ROOT / "weights").is_dir() else "FAIL", "weights/")
    for name, want in STARTER_KIT.items():
        p = ROOT / name
        if p.exists():
            same = sha256(p) == want
            report("OK" if same else "FAIL", f"{name} без изменений",
                   "" if same else "файл отличается от стартового набора — верните оригинал")


def check_requirements():
    print("\n[2] Зависимости (pip install -r requirements.txt на чистой машине)")
    lines = [l.split("#")[0].strip() for l in (ROOT / "requirements.txt").read_text().splitlines()]
    reqs = {re.split(r"[<>=!~ ;\[]", l, 1)[0].lower(): l for l in lines if l}
    for pkg in ("torch", "torchvision", "ultralytics"):
        spec = reqs.get(pkg, "")
        report("OK" if "==" in spec else "FAIL", f"{pkg} закреплён точно", spec or "не указан")
    report("OK" if "lap" in reqs else "FAIL", "lap в requirements", "нужен ByteTrack офлайн")
    has_headless = "opencv-python-headless" in reqs
    report("FAIL" if has_headless else "OK", "нет opencv-python-headless",
           "конфликтует с opencv-python из ultralytics" if has_headless else "")
    try:
        import torch
        tv = torch.__version__.split("+")[0]
        want = reqs.get("torch", "").split("==")[-1]
        report("OK" if tv == want else "WARN", "локальный torch совпадает с requirements",
               f"локально {torch.__version__}, в requirements {want}")
    except ImportError:
        report("WARN", "torch не установлен локально")


def check_weights():
    print("\n[3] Веса (<= 5 ГБ, офлайн)")
    total = 0
    for name, want in WEIGHTS.items():
        p = ROOT / "weights" / name
        if not p.exists():
            report("WARN" if (ROOT / "weights" / "download.sh").exists() else "FAIL", f"weights/{name}",
                   "нет в репозитории — организаторам придётся запускать download.sh; надёжнее закоммитить")
            continue
        total += p.stat().st_size
        report("OK" if sha256(p) == want else "FAIL", f"weights/{name} sha256")
    report("OK" if total <= MAX_WEIGHTS_BYTES else "FAIL", f"размер весов {total / 1e6:.0f} МБ")
    for f in ("zones_ref.jpg", "zones_ref.json"):
        ok = (ROOT / f).exists()
        report("OK" if ok else "FAIL", f, "" if ok else
               "нет опорного кадра — зоны не совмещаются с видео (tools/make_zone_ref.py)")


def check_solution():
    print("\n[4] Интерфейс solution.py")
    try:
        import solution
        from evaluate import OFFICIAL_CLASSES
    except Exception as exc:
        report("FAIL", "import solution", repr(exc))
        return
    extra = [c for c in solution.CLASSES if c not in OFFICIAL_CLASSES]
    report("OK" if not extra else "FAIL", "CLASSES ⊆ официальных", f"лишние: {extra}" if extra else
           f"{len(solution.CLASSES)} классов: {solution.CLASSES}")
    for name in ("detect_events", "RiskEstimator"):
        report("OK" if hasattr(solution, name) else "FAIL", f"solution.{name}")
    rs = solution.RiskEstimator
    report("OK" if all(hasattr(rs, m) for m in ("reset", "step")) else "FAIL", "RiskEstimator.reset/step")


def check_predictions():
    print("\n[5] predictions_samples.json")
    p = ROOT / "predictions_samples.json"
    if not p.exists():
        report("FAIL", "файл", "python run_submission.py --videos samples --out predictions_samples.json --team <команда>")
        return
    pred = json.loads(p.read_text())
    import evaluate
    errors, warnings = evaluate.validate(pred)
    report("OK" if not errors else "FAIL", "evaluate.py --validate-only",
           f"{len(errors)} ошибок: {errors[:3]}" if errors else f"{len(warnings)} предупреждений")
    team = pred.get("team", "")
    report("OK" if team and team != "unnamed-team" else "FAIL", "имя команды", team or "пусто")
    samples = ROOT / "samples"
    if samples.is_dir():
        vids = {v.name for v in samples.iterdir() if v.suffix.lower() == ".mp4"}
        missing = sorted(vids - set(pred.get("videos", {})))
        report("OK" if not missing else "FAIL", "все сэмплы в файле", f"нет: {missing}" if missing else
               f"{len(vids)} видео")
    for vid, log in pred.get("log", {}).items():
        errs = log.get("errors", [])
        report("OK" if not errs else "FAIL", f"{vid}: лог харнесса", "; ".join(e.splitlines()[0] for e in errs[:2]))
        total, budget = log.get("total_sec"), log.get("budget_sec")
        if total and budget:
            frac = total / budget
            report("OK" if frac <= TIME_MARGIN else "WARN", f"{vid}: время",
                   f"{total:.0f}s из {budget:.0f}s ({frac:.0%} бюджета, цель <= {TIME_MARGIN:.0%})")
        if "part_b_sec" not in log:
            report("FAIL", f"{vid}: Part B", "risk не считался — прогон был с --no-risk?")
    for vid, entry in pred.get("videos", {}).items():
        if not entry.get("risk"):
            report("WARN", f"{vid}: пустая risk-кривая")
    if (ROOT / "zones_ref.json").exists() and git("rev-parse", "HEAD"):
        newer = git("log", "-1", "--format=%ct", "--", "src", "solution.py", "zones.json", "zones_ref.json")
        pred_t = git("log", "-1", "--format=%ct", "--", "predictions_samples.json")
        if newer and pred_t and int(newer) > int(pred_t):
            report("WARN", "predictions_samples.json старее кода",
                   "код/зоны менялись после последней генерации — перегенерируйте")


def check_readme():
    print("\n[6] README")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for key, pat in {"установка и запуск": r"pip install -r requirements\.txt",
                     "как получить веса": r"download\.sh",
                     "подход": r"(?i)approach",
                     "датасеты и лицензии": r"(?i)licen[cs]e",
                     "сиды/недетерминизм": r"(?i)determinis|seed",
                     "команда": r"(?i)team"}.items():
        report("OK" if re.search(pat, text) else "FAIL", f"README: {key}")
    placeholders = re.findall(r"<team name>|<[^>]*команд[^>]*>|\| … \|", text)
    report("OK" if not placeholders else "FAIL", "README без заглушек",
           f"заполните: {sorted(set(placeholders))}" if placeholders else "")


def check_git():
    print("\n[7] Git")
    if git("rev-parse", "--is-inside-work-tree") != "true":
        report("WARN", "это не git-репозиторий (git init && git add . && git commit)")
        return
    dirty = git("status", "--porcelain")
    report("OK" if not dirty else "WARN", "всё закоммичено", f"{len(dirty.splitlines())} изменённых файлов" if dirty else "")
    tracked = (git("ls-files") or "").splitlines()
    for f in ("zones_ref.jpg", "zones.json", "weights/yolo11s.pt", "weights/yolo11n.pt", "predictions_samples.json"):
        if (ROOT / f).exists():
            report("OK" if f in tracked else "WARN", f"{f} в git", "" if f in tracked else "файл есть, но не добавлен")
    videos = [f for f in tracked if f.lower().endswith(".mp4")]
    report("OK" if not videos else "WARN", "видео не закоммичены", f"{videos[:3]}" if videos else "")
    tag = git("describe", "--tags", "--exact-match")
    report("OK" if tag else "WARN", "тег на текущем коммите", tag or "git tag v1.0 && git push --tags")
    remote = git("remote", "get-url", "origin")
    report("OK" if remote else "WARN", "remote origin", remote or "нет — репозиторий должен быть публичным")


def check_determinism(video: str):
    print(f"\n[8] Детерминизм: два прогона харнесса на {video}")
    outs = []
    for k in (1, 2):
        out = ROOT / "debug" / f"_det_{k}.json"
        out.parent.mkdir(exist_ok=True)
        r = subprocess.run([sys.executable, "run_submission.py", "--videos", video, "--out", str(out),
                            "--team", "det"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            report("FAIL", f"прогон {k}", r.stderr.strip().splitlines()[-1] if r.stderr else "")
            return
        outs.append(json.loads(out.read_text())["videos"])
    a, b = outs
    same_ev = all(a[v]["events"] == b[v]["events"] for v in a)
    report("OK" if same_ev else "FAIL", "events совпадают")
    diff = max((abs(x[1] - y[1]) for v in a for x, y in zip(a[v]["risk"], b[v]["risk"])), default=0.0)
    report("OK" if diff <= 1e-3 else "WARN", "risk совпадает", f"макс. расхождение {diff:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--determinism", default=None, help="видео для двойного прогона (самое короткое)")
    args = ap.parse_args()
    check_layout()
    check_requirements()
    check_weights()
    check_solution()
    check_predictions()
    check_readme()
    check_git()
    if args.determinism:
        check_determinism(args.determinism)
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"\nИтог: {len(fails)} FAIL, {len(warns)} WARN")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
